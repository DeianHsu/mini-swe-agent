"""离线验证显式网络重试机制与本地仓库一键沙箱修复入口。"""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx2
from openai import APIConnectionError, BadRequestError, OpenAI

from swe import cost, fix_repo, mini_agent, use_llm


class TransientRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swe-retry-offline-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.attempts = 0
        self.outcomes = []
        self.sleep_delays = []
        self.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        }
        self.client = OpenAI(
            api_key="offline-test-key",
            base_url="https://example.invalid",
            timeout=120.0,
            max_retries=0,
            http_client=httpx2.Client(
                transport=httpx2.MockTransport(self._respond)
            ),
        )
        self.addCleanup(self.client.close)

    def _respond(self, request):
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _ok_response(self, content="finish"):
        return httpx2.Response(
            200,
            json={
                "id": "resp-ok",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": self.usage,
            },
        )

    def test_transient_connection_reset_recovers_and_logs_retry_events(self):
        self.outcomes = [
            httpx2.ConnectError("Connection reset by peer"),
            httpx2.ConnectError("Connection reset by peer"),
            self._ok_response(),
        ]
        trace_path = self.root / "trajectory.jsonl"

        with (
            patch.object(mini_agent, "create_client", return_value=self.client),
            patch.object(use_llm.time, "sleep", side_effect=self.sleep_delays.append),
        ):
            result = mini_agent.run_agent("offline check", trace_path=trace_path)

        self.assertEqual(self.attempts, 3)
        self.assertEqual(self.sleep_delays, [1.0, 2.0])
        self.assertEqual(result["exit_reason"], "finished")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["transient_retries"], 2)

        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        retries = [event["data"] for event in events if event["kind"] == "retry"]
        self.assertEqual(len(retries), 2)
        self.assertEqual([item["attempt"] for item in retries], [1, 2])
        self.assertEqual(retries[0]["error_type"], "APIConnectionError")

        usage = cost.read_run_usage(trace_path)
        self.assertTrue(usage["usage_complete"])
        self.assertEqual(usage["model_calls"], 1)
        self.assertEqual(usage["accounted_calls"], 1)

    def test_non_retryable_client_error_fails_immediately(self):
        self.outcomes = [
            httpx2.Response(
                400, json={"error": {"message": "invalid request payload"}}
            )
        ]
        with patch.object(use_llm.time, "sleep", side_effect=self.sleep_delays.append):
            with self.assertRaises(BadRequestError):
                use_llm.send_messages(
                    self.client, [{"role": "user", "content": "bad"}]
                )
        self.assertEqual(self.attempts, 1)
        self.assertEqual(self.sleep_delays, [])

    def test_exhausted_retries_raises_and_marks_usage_incomplete(self):
        self.outcomes = [
            httpx2.ConnectError("Connection reset by peer") for _ in range(4)
        ]
        trace_path = self.root / "exhausted.jsonl"

        with (
            patch.object(mini_agent, "create_client", return_value=self.client),
            patch.object(use_llm.time, "sleep", side_effect=self.sleep_delays.append),
        ):
            with self.assertRaises(APIConnectionError):
                mini_agent.run_agent("offline check", trace_path=trace_path)

        self.assertEqual(self.attempts, 4)
        self.assertEqual(self.sleep_delays, [1.0, 2.0, 4.0])
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            sum(1 for event in events if event["kind"] == "retry"), 3
        )
        self.assertEqual(
            sum(1 for event in events if event["kind"] == "error"), 1
        )
        usage = cost.read_run_usage(trace_path)
        self.assertFalse(usage["usage_complete"])


class FixRepoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swe-fix-repo-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "sample_repo"
        (self.repo / ".git").mkdir(parents=True)
        (self.repo / "app.py").write_text("print('bug')\n", encoding="utf-8")

    def test_fix_local_repo_exports_patch_and_removes_container(self):
        docker_calls = []

        def fake_run(command, **kwargs):
            docker_calls.append(command)
            if command[:2] == ["docker", "exec"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="diff --git a/app.py b/app.py\n+fixed\n", stderr=""
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        output_dir = self.root / "out"
        with (
            patch.object(fix_repo.subprocess, "run", side_effect=fake_run),
            patch.object(
                fix_repo.budget,
                "run_with_budget",
                return_value={
                    "exit_reason": "finished",
                    "model_calls": 3,
                    "transient_retries": 1,
                    "known_cost_usd": "0.012",
                    "seconds": 9.5,
                },
            ) as mock_budget,
        ):
            summary = fix_repo.fix_local_repo(
                self.repo, "AssertionError in app.py", output_dir
            )

        self.assertEqual(summary["exit_reason"], "finished")
        self.assertEqual(summary["model_calls"], 3)
        self.assertEqual(summary["transient_retries"], 1)
        patch_path = output_dir / "fix.patch"
        self.assertEqual(
            patch_path.read_text(encoding="utf-8"),
            "diff --git a/app.py b/app.py\n+fixed\n",
        )
        self.assertTrue((output_dir / "fix_summary.json").is_file())
        self.assertIn("--network", docker_calls[0])
        self.assertIn("none", docker_calls[0])
        self.assertEqual(docker_calls[1][:2], ["docker", "cp"])
        self.assertEqual(docker_calls[-1][:3], ["docker", "rm", "-f"])
        mock_budget.assert_called_once()

    def test_fix_local_repo_cleans_up_container_on_agent_failure(self):
        docker_calls = []

        def fake_run(command, **kwargs):
            docker_calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with (
            patch.object(fix_repo.subprocess, "run", side_effect=fake_run),
            patch.object(
                fix_repo.budget,
                "run_with_budget",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                fix_repo.fix_local_repo(
                    self.repo, "crash on startup", self.root / "failed_out"
                )

        self.assertEqual(docker_calls[-1][:3], ["docker", "rm", "-f"])


if __name__ == "__main__":
    unittest.main()
