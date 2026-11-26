"""离线验证思考模式的请求、工具历史回传和预算进程传参，不访问真实模型。"""

from decimal import Decimal
import json
import os
from pathlib import Path
from runpy import run_path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx2
from openai import OpenAI

from swe import budget, cost, mini_agent, use_llm


class ThinkingModeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swe-thinking-offline-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = []
        self.responses = []
        self.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 200,
            "prompt_cache_hit_tokens": 20,
            "prompt_cache_miss_tokens": 80,
            "completion_tokens_details": {"reasoning_tokens": 60},
        }
        # 保留真实 SDK 的请求序列化和响应解析，只将 HTTP 传输替换为内存响应。
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
        client_patch = patch.object(
            mini_agent, "create_client", return_value=self.client
        )
        self.create_client = client_patch.start()
        self.addCleanup(client_patch.stop)

    def _respond(self, request):
        self.requests.append(json.loads(request.content))
        self.assertEqual(request.extensions["timeout"]["read"], 120.0)
        return httpx2.Response(200, json=self.responses.pop(0))

    def _response(self, message, finish_reason="stop"):
        return {
            "id": "offline-response",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": self.usage,
        }

    def test_default_request_still_explicitly_disables_thinking(self):
        self.responses.append(
            self._response({"role": "assistant", "content": "finish"})
        )
        _, metadata = use_llm.send_messages(
            self.client, [{"role": "user", "content": "offline check"}]
        )
        self.assertEqual(self.requests[0]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", self.requests[0])
        self.assertEqual(metadata["thinking_mode"], "disabled")

    def test_missing_deepseek_configuration_fails_before_creating_client(self):
        settings = {
            "DEEPSEEK_API_KEY": "offline-test-key",
            "DEEPSEEK_BASE_URL": "https://example.invalid",
            "OPENAI_API_KEY": "other-offline-key",
            "OPENAI_BASE_URL": "https://other.example.invalid",
        }
        for missing in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
            with (
                self.subTest(missing=missing),
                patch.dict(os.environ, settings),
                patch.object(use_llm, "load_dotenv"),
                patch.object(use_llm, "OpenAI") as client,
            ):
                del os.environ[missing]
                with self.assertRaisesRegex(ValueError, missing):
                    use_llm.create_client()
                client.assert_not_called()

    def test_import_does_not_load_model_config_or_open_client(self):
        with (
            patch("dotenv.load_dotenv") as load_config,
            patch("openai.OpenAI") as client,
        ):
            run_path(use_llm.__file__)
        load_config.assert_not_called()
        client.assert_not_called()

    def test_max_thinking_survives_tool_round_trip_and_counts_output_once(
        self,
    ):
        reasoning = "mock reasoning " * 20000
        self.responses.extend(
            [
                self._response(
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": reasoning,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
                self._response(
                    {
                        "role": "assistant",
                        "content": "finish",
                        "reasoning_content": "done",
                    }
                ),
            ]
        )
        trace_path = self.root / "trajectory.jsonl"
        with patch.object(
            mini_agent, "execute", return_value="/app"
        ) as execute:
            result = mini_agent.run_agent(
                "offline check", thinking_mode="max", trace_path=trace_path
            )
        execute.assert_called_once_with(
            "pwd",
            container_name="mini-swe-learning",
            workdir="/workspace",
            timeout=60,
        )
        self.create_client.assert_called_once_with()
        self.assertTrue(self.client.is_closed())
        self.assertEqual(result["thinking_mode"], "max")
        self.assertEqual(result["model_calls"], 2)
        for request in self.requests:
            self.assertEqual(request["thinking"], {"type": "enabled"})
            self.assertEqual(request["reasoning_effort"], "max")
        previous = next(
            message
            for message in self.requests[1]["messages"]
            if message["role"] == "assistant"
        )
        self.assertEqual(previous["reasoning_content"], reasoning)
        tool_reply = next(
            message
            for message in self.requests[1]["messages"]
            if message["role"] == "tool"
        )
        self.assertEqual(
            tool_reply,
            {"role": "tool", "tool_call_id": "call-1", "content": "/app"},
        )
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        for event in events:
            if event["kind"] in ("start", "assistant", "end"):
                self.assertEqual(event["data"]["thinking_mode"], "max")
        usage = cost.read_run_usage(trace_path)
        self.assertTrue(usage["usage_complete"])
        self.assertEqual(usage["completion_tokens"], 200)
        self.assertEqual(usage["total_tokens"], 400)
        self.assertEqual(
            Decimal(usage["known_cost_usd"]), Decimal("0.00028824")
        )

    def test_invalid_prompt_usage_preserves_other_responses(self):
        trace_path = self.root / "trajectory.jsonl"
        mini_agent._record_event(
            trace_path,
            "start",
            {
                "token_prices_per_million": {
                    key: str(price)
                    for key, price in cost.TOKEN_PRICES_PER_MILLION.items()
                },
            },
        )
        for step, prompt_tokens in enumerate((100, None, 100), start=1):
            mini_agent._record_event(trace_path, "request", {"step": step})
            mini_agent._record_event(
                trace_path,
                "assistant",
                {
                    "step": step,
                    "usage": {**self.usage, "prompt_tokens": prompt_tokens},
                },
            )

        usage = cost.read_run_usage(trace_path)
        self.assertEqual(
            (usage["model_calls"], usage["accounted_calls"]), (3, 2)
        )
        self.assertEqual(usage["total_tokens"], 400)
        self.assertEqual(
            Decimal(usage["known_cost_usd"]), Decimal("0.00028824")
        )
        self.assertFalse(usage["usage_complete"])
        self.assertIsNone(usage["budget_cost_usd"])

    def test_unsupported_mode_fails_before_sending_a_request(self):
        with self.assertRaisesRegex(ValueError, "thinking_mode"):
            use_llm.send_messages(self.client, [], thinking_mode="high")
        self.assertEqual(self.requests, [])

    def test_worker_uses_the_mode_saved_in_case_json(self):
        self.responses.append(
            self._response(
                {
                    "role": "assistant",
                    "content": "finish",
                    "reasoning_content": "done",
                }
            )
        )
        process = MagicMock(returncode=0)
        process.wait.return_value = 0
        process.poll.return_value = 0

        def start_worker(command, **kwargs):
            budget._run_worker(Path(command[-1]))
            return process

        output = self.root / "case"
        with (
            patch.object(budget.subprocess, "Popen", side_effect=start_worker),
            patch.object(
                budget.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            result = budget.run_with_budget(
                "offline check", "unused", "/app", output, thinking_mode="max"
            )
        self.assertEqual(self.requests[0]["reasoning_effort"], "max")
        self.assertEqual(result["thinking_mode"], "max")
        self.assertTrue(result["usage_complete"])
        settings = json.loads(
            (output / "case.json").read_text(encoding="utf-8")
        )
        self.assertEqual(settings["thinking_mode"], "max")
        self.assertEqual(
            (
                settings["max_steps"],
                settings["command_timeout"],
                settings["cost_limit_usd"],
                settings["task_timeout_seconds"],
            ),
            (250, 60, "3", 3000),
        )

    def test_timeout_preserves_mode_without_inventing_model_usage(self):
        process = MagicMock(returncode=-9)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("worker", 3000),
            -9,
        ]
        process.poll.return_value = None
        with (
            patch.object(budget.subprocess, "Popen", return_value=process),
            patch.object(
                budget.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            result = budget.run_with_budget(
                "offline check",
                "unused",
                "/app",
                self.root / "timeout",
                thinking_mode="max",
            )
        process.kill.assert_called_once()
        self.assertEqual(result["exit_reason"], "time_limit")
        self.assertEqual(result["thinking_mode"], "max")
        self.assertIsNone(result["budget_cost_usd"])
        self.assertEqual(self.requests, [])

    def test_worker_launch_failure_stops_container_and_saves_error(self):
        output = self.root / "failed"
        with (
            patch.object(
                budget.subprocess,
                "Popen",
                side_effect=OSError("worker launch failed"),
            ),
            patch.object(
                budget.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as stop,
        ):
            with self.assertRaisesRegex(OSError, "worker launch failed"):
                budget.run_with_budget(
                    "offline check", "unused", "/app", output
                )

        stop.assert_called_once()
        self.assertEqual(
            stop.call_args.args[0], ["docker", "stop", "--time", "0", "unused"]
        )
        result = json.loads(
            (output / "agent_result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["exit_reason"], "error")
        self.assertEqual(result["error"], "worker launch failed")
        self.assertEqual(result["model_calls"], 0)

    def test_invalid_reply_keeps_prior_cost_and_worker_error(self):
        self.responses.extend(
            [
                self._response(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
                {
                    **self._response(
                        {"role": "assistant", "content": "finish"}
                    ),
                    "usage": {**self.usage, "prompt_tokens": None},
                },
            ]
        )
        process = MagicMock(returncode=1)
        process.poll.return_value = 1

        def start_worker(command, **kwargs):
            with self.assertRaisesRegex(ValueError, "prompt_tokens"):
                budget._run_worker(Path(command[-1]))
            return process

        run_dir = self.root / "bad-usage"
        with (
            patch.object(mini_agent, "execute", return_value="/app"),
            patch.object(budget.subprocess, "Popen", side_effect=start_worker),
            patch.object(
                budget.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            result = budget.run_with_budget(
                "offline check", "unused", "/app", run_dir
            )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            (
                result["exit_reason"],
                result["error_type"],
                result["worker_returncode"],
            ),
            ("error", "ValueError", 1),
        )
        self.assertIn("prompt_tokens", result["error"])
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["budget_cost_usd"])
        self.assertEqual(result["accounted_calls"], 1)
        self.assertEqual(
            Decimal(result["known_cost_usd"]), Decimal("0.00014412")
        )
        self.assertEqual(
            json.loads(
                (run_dir / "agent_result.json").read_text(encoding="utf-8")
            ),
            result,
        )
        self.assertTrue(self.client.is_closed())

    def test_worker_exit_code_is_saved_without_worker_result(self):
        process = MagicMock(returncode=9)
        process.poll.return_value = 9
        with (
            patch.object(budget.subprocess, "Popen", return_value=process),
            patch.object(
                budget.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            result = budget.run_with_budget(
                "offline check", "unused", "/app", self.root / "worker-exit"
            )
        self.assertEqual(
            (result["exit_reason"], result["worker_returncode"]), ("error", 9)
        )

    def test_budget_cli_defaults_to_disabled_and_accepts_max(self):
        task = self.root / "task.txt"
        task.write_text("offline check", encoding="utf-8")
        for mode in ("disabled", "max"):
            arguments = [
                "budget",
                "--task-file",
                str(task),
                "--container",
                "unused",
                "--output",
                str(self.root / mode),
            ]
            if mode == "max":
                arguments.extend(["--thinking-mode", "max"])
            with (
                self.subTest(mode=mode),
                patch.object(budget.sys, "argv", arguments),
                patch.object(
                    budget,
                    "run_with_budget",
                    return_value={"exit_reason": "finished"},
                ) as run,
                patch("builtins.print"),
            ):
                budget.main()
            self.assertEqual(run.call_args.kwargs["thinking_mode"], mode)


if __name__ == "__main__":
    unittest.main()
