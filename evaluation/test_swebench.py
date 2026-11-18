"""评测层离线回归；不调用模型，不创建 Docker 资源，不依赖主线内部接口。"""

import ast
from decimal import Decimal
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import report
import swebench
import verifier


TASK_CONFIG = '''
[agent]
network_mode = "no-network"
timeout_sec = 3000
[environment]
docker_image = "example.invalid/image:task"
build_timeout_sec = 1800
cpus = 1
memory_mb = 4096
[verifier]
timeout_sec = 3000
'''


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swebench-offline-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.instance_id = "instance_example__repo-abc"
        task = self.dataset / "v2/tasks" / self.instance_id
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text(TASK_CONFIG, encoding="utf-8")
        (task / "instruction.md").write_text("Fix the example.", encoding="utf-8")
        (task / "tests/test.sh").write_text("exit 0\n", encoding="utf-8")
        (task / "tests/config.json").write_text("{}", encoding="utf-8")
        with patch.object(swebench, "TASK_COUNT", 1):
            self.manifest = swebench.make_manifest(self.dataset, "run-1", "commit-1")
        self.output = self.root / "output"
        self.metrics = {
            "exit_reason": "finished", "model_calls": 2, "accounted_calls": 2,
            "seconds": 12.5, "prompt_tokens": 200, "completion_tokens": 30,
            "total_tokens": 230, "known_cost_usd": "0.000096",
            "budget_cost_usd": "0.000096", "usage_complete": True,
        }
        self.commands = []
        self.reward = "1"
        self.verifier_timeout = False
        self.setup_failure = False

    def command(self, args, log_path, *, timeout=120, output=None, check=True):
        self.commands.append(args)
        if args[:2] == ["docker", "pull"] and self.setup_failure:
            raise subprocess.CalledProcessError(1, args)
        stdout = b""
        returncode = 0
        if args[:3] == ["docker", "image", "inspect"]:
            stdout = b'[{"Id":"sha256:pristine"}]'
        elif verifier.PROBE_SCRIPT in args:
            stdout = b"/app\nbase-commit\n"
        elif swebench.CAPTURE_SCRIPT in args:
            output.write(b"diff --git a/code b/code\n")
        elif "/tests/test.sh" in args:
            if self.verifier_timeout:
                raise subprocess.TimeoutExpired(args, timeout)
            verifier_logs = self.output / self.instance_id / "verifier"
            (verifier_logs / "reward.txt").write_text(self.reward, encoding="utf-8")
            returncode = 0 if self.reward == "1" else 1
        return subprocess.CompletedProcess(args, returncode, stdout)

    def agent(self, task_file, container, workdir, output, log_path, *, thinking_mode):
        output.mkdir()
        verifier.save_json(output / "agent_result.json", {**self.metrics, "thinking_mode": thinking_mode})
        verifier.save_json(output / "case.json", {**self.manifest["budget"], "thinking_mode": thinking_mode})
        return 1 if self.metrics["exit_reason"] == "error" else 0

    def evaluate(self):
        with patch.object(verifier, "run_command", side_effect=self.command), \
             patch.object(swebench, "run_budget_cli", side_effect=self.agent) as agent:
            result = swebench.eval_case(self.dataset, self.manifest, 0, self.output)
        return result, agent

    def test_full_manifest_rejects_missing_tasks(self):
        with self.assertRaisesRegex(ValueError, "642"):
            swebench.make_manifest(self.dataset, "run", "commit")

    def test_max_mode_keeps_budget_and_reaches_agent_and_reports(self):
        with patch.object(swebench, "TASK_COUNT", 1):
            thinking_manifest = swebench.make_manifest(self.dataset, "run-1", "commit-1", thinking_mode="max")
        self.assertEqual(self.manifest["thinking_mode"], "disabled")
        self.assertEqual(thinking_manifest["budget"], self.manifest["budget"])
        self.assertEqual(thinking_manifest["instances"], self.manifest["instances"])
        self.manifest = thinking_manifest
        result, agent = self.evaluate()
        self.assertEqual(agent.call_args.kwargs["thinking_mode"], "max")
        self.assertEqual(result["thinking_mode"], "max")
        self.assertIs(result["resolved"], True)
        summary = report.write_report(self.output, manifest=self.manifest)
        self.assertEqual(summary["thinking_mode"], "max")
        self.assertIn("最高（max）", (self.output / "report.md").read_text(encoding="utf-8"))
        with (self.output / "report.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["thinking_mode"], "max")

    def test_max_mode_is_forwarded_to_the_budget_command(self):
        process = MagicMock(returncode=0)
        process.wait.return_value = 0
        with patch.object(swebench.subprocess, "Popen", return_value=process) as popen:
            swebench.run_budget_cli(self.root / "instruction.md", "container", "/app", self.output,
                                    self.root / "budget.log", thinking_mode="max")
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--thinking-mode") + 1], "max")
        self.assertIn("swe.budget", command)

    def test_three_batches_cover_exactly_the_full_set(self):
        indices = [i for batch in range(3) for i in range(
            batch * swebench.BATCH_SIZE, (batch + 1) * swebench.BATCH_SIZE)]
        self.assertEqual(indices, list(range(swebench.TASK_COUNT)))
        self.assertLessEqual(swebench.BATCH_SIZE, 256)

    def test_solve_offline_and_grade_in_fresh_container(self):
        result, agent = self.evaluate()
        self.assertEqual(result["status"], "graded")
        self.assertIs(result["resolved"], True)
        self.assertEqual(result["seconds"], 12.5)
        self.assertEqual(result["total_tokens"], 230)
        agent.assert_called_once()
        solving = next(args for args in self.commands if "-d" in args and "none" in args)
        verifying = next(args for args in self.commands if "bridge" in args)
        capture = next(args for args in self.commands if swebench.CAPTURE_SCRIPT in args)
        self.assertNotIn("/tests", " ".join(solving))
        self.assertIn("readonly", " ".join(capture))
        self.assertNotIn("type=volume", " ".join(verifying))
        self.assertNotEqual(solving[solving.index("--name") + 1], verifying[verifying.index("--name") + 1])
        for command in (solving, capture, verifying):
            self.assertIn("sha256:pristine", command)
            self.assertIn("4096m", command)
            self.assertNotIn("DEEPSEEK_API_KEY", " ".join(command))

    def test_failed_tests_are_a_valid_negative_verdict(self):
        self.reward = "0"
        result, agent = self.evaluate()
        self.assertEqual(result["status"], "graded")
        self.assertIs(result["resolved"], False)
        self.assertEqual(result["known_cost_usd"], self.metrics["known_cost_usd"])
        agent.assert_called_once()

    def test_agent_error_still_grades_partial_patch(self):
        self.metrics.update(exit_reason="error", usage_complete=False, budget_cost_usd=None)
        result, agent = self.evaluate()
        self.assertIs(result["resolved"], True)
        self.assertEqual(result["agent_returncode"], 1)
        self.assertEqual(result["known_cost_usd"], "0.000096")
        self.assertFalse(result["usage_complete"])
        agent.assert_called_once()

    def test_interrupted_budget_keeps_worker_cost(self):
        confirmed_cost = self.metrics["budget_cost_usd"]
        self.metrics = {key: self.metrics[key] for key in ("exit_reason", "model_calls", "budget_cost_usd")}

        def interrupted_budget(*args, **kwargs):
            self.agent(*args, **kwargs)
            return -9

        with patch.object(verifier, "run_command", side_effect=self.command), \
             patch.object(swebench, "run_budget_cli", side_effect=interrupted_budget):
            result = swebench.eval_case(self.dataset, self.manifest, 0, self.output)

        self.assertEqual(result["agent_returncode"], -9)
        self.assertIsNone(result["budget_cost_usd"])
        self.assertEqual(result["known_cost_usd"], confirmed_cost)
        summary = report.write_report(self.output, manifest=self.manifest)
        self.assertEqual(summary["known_total_cost_usd"], confirmed_cost)
        self.assertIsNone(summary["total_cost_usd"])

    def test_interrupted_budget_reads_trace_without_worker_result(self):
        agent_dir = self.output / self.instance_id / "agent"
        trace_path = agent_dir / "trajectory.jsonl"
        usage = {key: value for key, value in self.metrics.items() if key not in ("exit_reason", "seconds")}
        usage.update(model_calls=3, budget_cost_usd=None, usage_complete=False)

        def interrupted_budget(task_file, container, workdir, run_dir, log_path, *, thinking_mode):
            run_dir.mkdir()
            verifier.save_json(run_dir / "case.json", {**self.manifest["budget"], "thinking_mode": thinking_mode})
            trace_path.touch()
            return -9

        read_command = [sys.executable, "-B", "-m", "swe.cost"]

        def command(args, log_path, **kwargs):
            if args[:4] == read_command:
                return subprocess.CompletedProcess(args, 0, json.dumps(usage).encode("utf-8"))
            return self.command(args, log_path, **kwargs)

        with patch.object(verifier, "run_command", side_effect=command) as run_command, \
             patch.object(swebench, "run_budget_cli", side_effect=interrupted_budget):
            result = swebench.eval_case(self.dataset, self.manifest, 0, self.output)

        run_command.assert_any_call([*read_command, str(trace_path)], self.output / self.instance_id / "runner.log")
        self.assertTrue(result["resolved"])
        self.assertEqual(result["agent_returncode"], -9)
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["accounted_calls"], 2)
        self.assertEqual(result["total_tokens"], 230)
        self.assertIsNone(result["seconds"])
        self.assertEqual(result["known_cost_usd"], self.metrics["known_cost_usd"])
        self.assertFalse(result["usage_complete"])
        summary = report.write_report(self.output, manifest=self.manifest)
        self.assertEqual(summary["known_total_cost_usd"], self.metrics["known_cost_usd"])
        self.assertIsNone(summary["total_cost_usd"])

    def test_verifier_timeout_preserves_consumption_and_cleans_resources(self):
        self.verifier_timeout = True
        result, agent = self.evaluate()
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["resolved"])
        self.assertEqual(result["total_tokens"], 230)
        self.assertEqual(result["stage"], "verifier")
        self.assertTrue(any(args[:3] == ["docker", "volume", "rm"] for args in self.commands))
        agent.assert_called_once()

    def test_invalid_reward_is_not_a_negative_verdict(self):
        self.reward = "invalid"
        result, _ = self.evaluate()
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["resolved"])

    def test_budget_timeout_kills_process_group_without_retry(self):
        process = MagicMock(pid=123, returncode=-9)
        process.wait.side_effect = [subprocess.TimeoutExpired("agent", 3180), -9]
        with patch.object(swebench.subprocess, "Popen", return_value=process) as popen, \
             patch.object(swebench.os, "killpg", create=True) as killpg, \
             patch.object(swebench.signal, "SIGKILL", 9, create=True):
            code = swebench.run_budget_cli(self.root / "instruction.md", "container", "/app",
                                            self.output, self.root / "budget.log")
        self.assertEqual(code, -9)
        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(process.wait.call_args_list[0].kwargs, {"timeout": 3180})
        killpg.assert_called_once_with(123, 9)

    def test_patch_replay_uses_image_files_and_keeps_all_agent_changes(self):
        git = shutil.which("git")
        if git is None:
            self.skipTest("需要 Git")
        bash = (str(Path(git).parent.parent / "bin/bash.exe")
                if sys.platform == "win32" else shutil.which("bash"))
        if bash is None or not Path(bash).is_file():
            self.skipTest("需要 Bash")

        base = self.root / "base"
        base.mkdir()

        def git_command(directory, *args):
            return subprocess.run(
                [git, "-c", "safe.directory=*", "-c", "core.autocrlf=false",
                 "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                 "-c", "user.name=Evaluation Test", "-c", "user.email=test@example.invalid",
                 "-C", str(directory), *args],
                capture_output=True, check=True,
            )

        git_command(base, "init")
        (base / "code.txt").write_bytes(b"before\n")
        (base / "deleted.txt").write_bytes(b"removed later\n")
        (base / ".gitignore").write_bytes(b"ignored.txt\n")
        git_command(base, "add", "-A")
        git_command(base, "commit", "-m", "基础版本")
        # 官方镜像可能带有未提交的改动和未跟踪文件，判分容器也会自带同一份内容。
        (base / "code.txt").write_bytes(b"image build adjustment\n")
        (base / "image-unchanged.txt").write_bytes(b"already in image\n")
        (base / "image-removed.txt").write_bytes(b"remove this image file\n")
        (base / "appendonlydir").mkdir()
        (base / "appendonlydir/state.rdb").write_bytes(b"\x00original redis state\xff")
        candidate, verifier = self.root / "candidate", self.root / "verifier"
        shutil.copytree(base, candidate)
        shutil.copytree(base, verifier)
        (candidate / "code.txt").write_bytes(b"committed\n")
        (candidate / "deleted.txt").unlink()
        binary = b"\x00\xff\x01binary data\n"
        (candidate / "new.bin").write_bytes(binary)
        (candidate / "ignored.txt").write_bytes(b"build output\n")
        (candidate / "image-removed.txt").unlink()
        (candidate / "appendonlydir/state.rdb").write_bytes(b"\x00updated redis state\xfe")
        git_command(candidate, "add", "-A")
        git_command(candidate, "commit", "-m", "智能体自行提交")
        (candidate / "code.txt").write_bytes(b"final uncommitted change\n")

        captured = subprocess.run(
            [bash, "-c", swebench.CAPTURE_SCRIPT, "capture", base.as_posix(), candidate.as_posix()],
            cwd=base, capture_output=True, check=True,
        )
        patch_file = self.root / "model.patch"
        patch_file.write_bytes(captured.stdout)
        git_command(verifier, "apply", "--binary", str(patch_file))
        self.assertEqual((verifier / "code.txt").read_bytes(), b"final uncommitted change\n")
        self.assertEqual((verifier / "new.bin").read_bytes(), binary)
        self.assertFalse((verifier / "deleted.txt").exists())
        self.assertFalse((verifier / "ignored.txt").exists())
        self.assertFalse((verifier / "image-removed.txt").exists())
        self.assertEqual((verifier / "appendonlydir/state.rdb").read_bytes(), b"\x00updated redis state\xfe")
        self.assertEqual((verifier / "image-unchanged.txt").read_bytes(), b"already in image\n")
        self.assertNotIn(b"image-unchanged.txt", captured.stdout)

    def test_setup_failure_spends_no_model_tokens(self):
        self.setup_failure = True
        result, agent = self.evaluate()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(Decimal(result["budget_cost_usd"]), 0)
        agent.assert_not_called()

    def test_missing_container_limits_saves_setup_error(self):
        config = self.dataset / "v2/tasks" / self.instance_id / "task.toml"
        config.write_text(TASK_CONFIG.replace("cpus = 1\n", ""), encoding="utf-8")
        result, agent = self.evaluate()

        self.assertEqual((result["status"], result["stage"]), ("error", "setup"))
        self.assertIn("cpus", result["error"])
        self.assertIsNone(result["resolved"])
        result_path = self.output / self.instance_id / "agent_result.json"
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), result)
        self.assertEqual(self.commands, [])
        agent.assert_not_called()

    def test_existing_attempt_is_never_overwritten(self):
        self.evaluate()
        with patch.object(swebench, "run_budget_cli") as agent, self.assertRaises(FileExistsError):
            swebench.eval_case(self.dataset, self.manifest, 0, self.output)
        agent.assert_not_called()

    def test_budget_mismatch_is_not_published_as_a_valid_score(self):
        def wrong_budget(*args, **kwargs):
            result = self.agent(*args, **kwargs)
            verifier.save_json(args[3] / "case.json", {
                **self.manifest["budget"], "max_steps": 999, "thinking_mode": kwargs["thinking_mode"],
            })
            return result
        with patch.object(verifier, "run_command", side_effect=self.command), \
             patch.object(swebench, "run_budget_cli", side_effect=wrong_budget):
            result = swebench.eval_case(self.dataset, self.manifest, 0, self.output)
        self.assertIsNone(result["resolved"])
        self.assertEqual(result["status"], "error")

    def test_thinking_mode_mismatch_blocks_grading(self):
        self.manifest["thinking_mode"] = "max"

        def wrong_mode(*args, **kwargs):
            result = self.agent(*args, **kwargs)
            verifier.save_json(args[3] / "case.json", {**self.manifest["budget"], "thinking_mode": "disabled"})
            return result

        with patch.object(verifier, "run_command", side_effect=self.command), \
             patch.object(swebench, "run_budget_cli", side_effect=wrong_mode):
            result = swebench.eval_case(self.dataset, self.manifest, 0, self.output)
        self.assertEqual(result["status"], "error")
        self.assertIn("思考模式", result["error"])
        self.assertIsNone(result["resolved"])
        self.assertEqual(result["known_cost_usd"], self.metrics["known_cost_usd"])
        self.assertFalse(any("bridge" in args for args in self.commands))

    def test_missing_budget_record_preserves_cost_but_blocks_grading(self):
        def missing_budget(*args, **kwargs):
            result = self.agent(*args, **kwargs)
            (args[3] / "case.json").unlink()
            return result

        with patch.object(verifier, "run_command", side_effect=self.command), \
             patch.object(swebench, "run_budget_cli", side_effect=missing_budget):
            result = swebench.eval_case(self.dataset, self.manifest, 0, self.output)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "FileNotFoundError")
        self.assertIsNone(result["resolved"])
        self.assertEqual(result["total_tokens"], 230)
        self.assertEqual(result["known_cost_usd"], "0.000096")
        self.assertFalse(any("bridge" in args for args in self.commands))
        summary = report.write_report(self.output, manifest=self.manifest)
        self.assertIsNone(summary["pass_at_1"])
        self.assertEqual(summary["ungraded"], 1)

    def test_missing_artifact_stays_in_report(self):
        self.evaluate()
        manifest = {**self.manifest, "instances": [self.instance_id, "missing-task"]}
        summary = report.write_report(self.output, manifest=manifest)
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["ungraded"], 1)
        self.assertIsNone(summary["pass_at_1"])
        self.assertIsNone(summary["total_cost_usd"])
        self.assertEqual(summary["known_total_cost_usd"], "0.000096")
        self.assertEqual(summary["cases"][1]["status"], "missing_result")
        for name in ("report.json", "report.csv", "report.md"):
            self.assertTrue((self.output / name).is_file())

    def test_report_includes_failed_task_cost(self):
        self.reward = "0"
        self.evaluate()
        summary = report.write_report(self.output, manifest=self.manifest)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["pass_at_1"], 0)
        self.assertEqual(summary["total_cost_usd"], "0.000096")

    def test_report_rejects_another_run(self):
        self.evaluate()
        with self.assertRaises(ValueError):
            report.write_report(self.output, manifest={**self.manifest, "run_id": "another-run"})

    def test_older_manifest_is_not_labeled_with_an_inferred_mode(self):
        self.evaluate()
        old_manifest = {key: value for key, value in self.manifest.items() if key != "thinking_mode"}
        summary = report.write_report(self.output, manifest=old_manifest)
        self.assertIsNone(summary["thinking_mode"])
        self.assertIn("思考模式 | 未记录", (self.output / "report.md").read_text(encoding="utf-8"))

    def test_evaluation_does_not_import_main_project(self):
        for module in (swebench, report, verifier):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[0], "swe")
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.split(".")[0] == "swe" for alias in node.names))


if __name__ == "__main__":
    unittest.main()
