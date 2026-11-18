"""校准入口的离线回归；模拟 Docker 命令，不拉取镜像或调用模型。"""

import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import calibrate
import report
import verifier


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swebench-calibration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.instance_id = "instance_example__repo-abc"
        self.task_dir = self.dataset / "v2/tasks" / self.instance_id
        (self.task_dir / "tests").mkdir(parents=True)
        (self.task_dir / "solution").mkdir()
        (self.task_dir / "task.toml").write_text(
            '[environment]\ndocker_image = "example.invalid/image:task"\n'
            'build_timeout_sec = 1800\ncpus = 1\nmemory_mb = 4096\n'
            '[verifier]\ntimeout_sec = 3000\n', encoding="utf-8",
        )
        (self.task_dir / "tests/test.sh").write_text("exit 0\n", encoding="utf-8")
        self.gold_patch = b"diff --git a/code b/code\nreference patch\n"
        (self.task_dir / "solution/gold_patch.diff").write_bytes(self.gold_patch)
        self.output = self.root / "calibration"
        self.commands = []
        self.rewards = {"empty": "0", "gold": "1"}
        self.timeout_mode = None
        self.patch_failure = False
        self.setup_failure = False
        self.cleanup_failure = False
        self.returncode = None

    def command(self, args, log_path, *, timeout=120, output=None, check=True):
        self.commands.append(args)
        stdout = b""
        returncode = 0
        if args[:2] == ["docker", "pull"] and self.setup_failure:
            raise subprocess.CalledProcessError(1, args)
        if args[:3] == ["docker", "image", "inspect"]:
            stdout = b'[{"Id":"sha256:pristine"}]'
        elif verifier.PROBE_SCRIPT in args:
            stdout = b"/app\nbase-commit\n"
        elif "apply" in args and self.patch_failure:
            raise subprocess.CalledProcessError(1, args)
        elif "/tests/test.sh" in args:
            mode = log_path.parent.name
            if mode == self.timeout_mode:
                raise subprocess.TimeoutExpired(args, timeout)
            returncode = (0 if self.rewards[mode] == "1" else 1) if self.returncode is None else self.returncode
            if returncode in (0, 1):
                (log_path.parent / "verifier/reward.txt").write_text(self.rewards[mode], encoding="utf-8")
        elif args[:3] == ["docker", "rm", "-f"] and check and self.cleanup_failure:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, returncode, stdout)

    def calibrate(self):
        with patch.object(verifier, "run_command", side_effect=self.command):
            return calibrate.run_calibration(self.dataset, self.instance_id, self.output)

    def test_pair_uses_fresh_containers_and_only_gold_receives_a_patch(self):
        result = self.calibrate()
        self.assertTrue(result["passed"])
        self.assertEqual([check["resolved"] for check in result["checks"]], [False, True])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["model_cost_usd"], "0")
        graders = [args for args in self.commands if "bridge" in args]
        self.assertEqual(len(graders), 2)
        names = [args[args.index("--name") + 1] for args in graders]
        self.assertNotEqual(names[0], names[1])
        # 第一种补丁完成清理后，才启动第二个判分容器。
        self.assertLess(self.commands.index(["docker", "rm", "-f", names[0]]), self.commands.index(graders[1]))
        for command in graders:
            self.assertIn("sha256:pristine", command)
            self.assertIn("4096m", command)
            self.assertNotIn("type=volume", " ".join(command))
        applications = [args for args in self.commands if "apply" in args]
        self.assertEqual(len(applications), 1)
        self.assertIn(names[1], applications[0])
        self.assertEqual(len([args for args in self.commands if args[:2] == ["docker", "pull"]]), 1)
        case_dir = self.output / self.instance_id
        self.assertEqual((case_dir / "empty/model.patch").read_bytes(), b"")
        self.assertEqual((case_dir / "gold/model.patch").read_bytes(), self.gold_patch)
        self.assertEqual(json.loads((case_dir / "calibration.json").read_text(encoding="utf-8")), result)
        self.assertTrue((case_dir / "calibration.md").is_file())
        self.assertFalse(list(self.output.rglob("agent_result.json")))
        with self.assertRaisesRegex(ValueError, "没有题目记录"):
            report.write_report(self.output)

    def test_unexpected_empty_success_and_gold_failure_both_fail_calibration(self):
        self.rewards = {"empty": "1", "gold": "0"}
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertEqual([check["status"] for check in result["checks"]], ["mismatch", "mismatch"])

    def test_timeout_is_not_an_expected_empty_failure_and_gold_still_runs(self):
        self.timeout_mode = "empty"
        result = self.calibrate()
        self.assertFalse(result["passed"])
        empty, gold = result["checks"]
        self.assertEqual(empty["status"], "error")
        self.assertIsNone(empty["resolved"])
        self.assertEqual(gold["status"], "passed")
        self.assertEqual(len([args for args in self.commands if args[:3] == ["docker", "rm", "-f"]]), 3)

    def test_invalid_reward_is_a_grading_error(self):
        self.rewards["empty"] = "invalid"
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["error_type"], "ValueError")
        self.assertIsNone(result["checks"][0]["resolved"])

    def test_verifier_process_error_is_not_a_valid_negative_score(self):
        self.returncode = 137
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertTrue(all(check["error_type"] == "RuntimeError" for check in result["checks"]))
        self.assertTrue(all("137" in check["error"] for check in result["checks"]))

    def test_reference_patch_application_failure_is_recorded(self):
        self.patch_failure = True
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][1]["error_type"], "CalledProcessError")
        self.assertIsNone(result["checks"][1]["resolved"])

    def test_missing_reference_patch_does_not_fall_back_to_empty(self):
        (self.task_dir / "solution/gold_patch.diff").unlink()
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][1]["error_type"], "FileNotFoundError")
        self.assertEqual(len([args for args in self.commands if "bridge" in args]), 1)

    def test_setup_failure_writes_a_failed_report(self):
        self.setup_failure = True
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["checks"], [])
        self.assertTrue((self.output / self.instance_id / "calibration.md").is_file())

    def test_cleanup_failure_does_not_report_calibration_success(self):
        self.cleanup_failure = True
        result = self.calibrate()
        self.assertFalse(result["passed"])
        self.assertTrue(all(check["cleanup_errors"] for check in result["checks"]))

    def test_previous_calibration_cannot_be_overwritten(self):
        self.calibrate()
        with self.assertRaises(FileExistsError):
            self.calibrate()

    def test_cli_fails_when_calibration_does_not_match(self):
        for passed in (True, False):
            with self.subTest(passed=passed), \
                 patch.object(calibrate.sys, "argv", ["calibrate.py", "--dataset", str(self.dataset),
                              "--instance-id", self.instance_id, "--output", str(self.output)]), \
                 patch.object(calibrate.sys, "platform", "linux"), \
                 patch.object(calibrate, "run_calibration", return_value={"passed": passed}), \
                 patch("builtins.print"):
                if passed:
                    calibrate.main()
                else:
                    with self.assertRaises(SystemExit) as error:
                        calibrate.main()
                    self.assertEqual(error.exception.code, 1)

    def test_calibration_does_not_import_the_full_evaluation_or_agent(self):
        for module in (calibrate, verifier):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(name.split(".")[0] in ("swebench", "swe") for name in imports))


if __name__ == "__main__":
    unittest.main()
