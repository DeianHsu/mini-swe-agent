"""离线验证 Docker Bash 命令执行、输出限制及镜像工具路径。"""

from io import BytesIO
import subprocess
import unittest
from unittest.mock import patch

from swe import environment


class DockerEnvironmentTests(unittest.TestCase):
    def test_output_limit_keeps_short_output_and_both_ends_of_large_output(
        self,
    ):
        short = "Exit code:0\nSTDOUT:\n正常输出\nSTDERR:\n"
        self.assertEqual(environment.limit_output(short), short)
        long = "start\n" + "中文🙂" * 100000 + "\nlast error"
        limited = environment.limit_output(long)
        self.assertLessEqual(len(limited), environment.MAX_OUTPUT_CHARS)
        self.assertTrue(limited.startswith("start\n"))
        self.assertTrue(limited.endswith("last error"))
        self.assertIn("输出已截断", limited)

    def test_execute_limits_large_output_and_keeps_exit_code_and_error(self):
        output_file, error_file = BytesIO(), BytesIO()

        def simulated_run(*, args, stdout, stderr, **kwargs):
            stdout.write(
                ("first line\r\n" + "🙂" * 100_000 + "\r\nlast line").encode(
                    "utf-8"
                )
            )
            stderr.write(b"real compile error")
            return subprocess.CompletedProcess(args, 1)

        with (
            patch.object(
                environment,
                "TemporaryFile",
                side_effect=[output_file, error_file],
            ),
            patch.object(output_file, "read", wraps=output_file.read) as read,
            patch.object(
                environment.subprocess, "run", side_effect=simulated_run
            ),
        ):
            output = environment.execute(
                "cat huge.log", container_name="offline", workdir="/app"
            )
        reads = [call.args[0] for call in read.call_args_list]
        self.assertEqual(len(reads), 2)
        self.assertTrue(
            all(0 < size <= 4 * environment.MAX_OUTPUT_CHARS for size in reads)
        )
        self.assertLessEqual(len(output), environment.MAX_OUTPUT_CHARS)
        self.assertTrue(
            output.startswith("Exit code:1\nSTDOUT:\nfirst line\n")
        )
        self.assertIn("last line\nSTDERR:\nreal compile error", output)
        self.assertIn("输出已截断", output)
        self.assertNotIn("\ufffd", output)

    def test_go_command_uses_the_image_environment(self):
        def simulated_shell(*, args, stdout, stderr, **kwargs):
            if "-lc" in args:
                stderr.write(b"bash: go: command not found")
                return subprocess.CompletedProcess(args, 127)
            stdout.write(b"go version go1.24.3 linux/amd64")
            return subprocess.CompletedProcess(args, 0)

        with patch.object(
            environment.subprocess, "run", side_effect=simulated_shell
        ):
            output = environment.execute(
                "go version", container_name="offline", workdir="/app"
            )
        self.assertIn("Exit code:0", output)
        self.assertIn("go1.24.3", output)


if __name__ == "__main__":
    unittest.main()
