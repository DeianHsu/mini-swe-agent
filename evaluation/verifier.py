"""评测层共用的官方镜像和补丁判分逻辑，不导入或调用智能体。"""

import json
from pathlib import Path
import shlex
import subprocess

DATASET_REPOSITORY = "scaleapi/SWE-bench_Pro-os"
DATASET_REVISION = "66f92766bba642462d4bbe5479e83f91f9211862"

PROBE_SCRIPT = r"""
set -euo pipefail
if git -c safe.directory='*' -C /app rev-parse --show-toplevel 2>/dev/null
then
    git -c safe.directory='*' -C /app rev-parse HEAD
else
    git -c safe.directory='*' -C /testbed rev-parse --show-toplevel
    git -c safe.directory='*' -C /testbed rev-parse HEAD
fi
"""


def save_json(path: Path, data: dict) -> None:
    """通过同目录临时文件替换结果，避免读到写入一半的 JSON。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_command(
    args: list[str],
    log_path: Path,
    *,
    timeout: float = 120,
    output=None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """命令和错误写入日志；大体积补丁或判分输出直接写入目标文件。"""
    with log_path.open("ab") as log:
        log.write(("\n$ " + shlex.join(args) + "\n").encode("utf-8"))
        log.flush()
        try:
            result = subprocess.run(
                args, stdout=subprocess.PIPE if output is None else output,
                stderr=log, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as error:
            if output is None and error.output:
                log.write(error.output)
            raise
        if output is None and result.stdout:
            log.write(result.stdout)
    if check:
        result.check_returncode()
    return result


def build_container_args(config: dict) -> list[str]:
    """根据题目配置生成 Docker 容器的共用启动参数。"""
    return [
        "--cpus",
        str(config["environment"]["cpus"]),
        "--memory",
        f'{config["environment"]["memory_mb"]}m',
        "--ulimit",
        "nofile=65536:65536",
        "--init",
        "--user",
        "0:0",
    ]


def prepare_image(
    config: dict, probe_name: str, log_path: Path
) -> tuple[str, str, str]:
    """拉取官方镜像并固定本地镜像 ID，读取干净仓库的路径和基础提交。"""
    image = config["environment"]["docker_image"]
    run_command(
        ["docker", "pull", "--platform", "linux/amd64", image],
        log_path,
        timeout=config["environment"]["build_timeout_sec"],
    )
    inspected = run_command(["docker", "image", "inspect", image], log_path)
    image_id = json.loads(inspected.stdout)[0]["Id"]
    probe = run_command(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            probe_name,
            "--network",
            "none",
            *build_container_args(config),
            "--entrypoint",
            "bash",
            image_id,
            "-c",
            PROBE_SCRIPT,
        ],
        log_path,
    )
    workdir, base_commit = probe.stdout.decode("utf-8").strip().splitlines()
    if workdir not in ("/app", "/testbed"):
        raise ValueError(f"不支持的任务工作目录：{workdir}")
    return image_id, workdir, base_commit


def verify_patch(task_dir: Path, config: dict, image_id: str, workdir: str,
                 patch_path: Path, case_dir: Path, container: str) -> dict:
    """在全新容器中只回放补丁并运行官方测试；调用者负责最终删除容器。"""
    log_path = case_dir / "runner.log"
    verifier_logs = case_dir / "verifier"
    verifier_logs.mkdir()
    # 不挂载解题工作卷；空补丁直接测试原始镜像，非空补丁严格检查能否应用。
    run_command(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--network",
            "bridge",
            *build_container_args(config),
            "--env",
            "SDL_AUDIODRIVER=dummy",
            "--mount",
            f"type=bind,source={(task_dir / 'tests').resolve()},"
            "target=/tests,readonly",
            "--mount",
            f"type=bind,source={verifier_logs},target=/logs/verifier",
            "--entrypoint",
            "sleep",
            image_id,
            "infinity",
        ],
        log_path,
    )
    if patch_path.stat().st_size:
        run_command(
            ["docker", "cp", str(patch_path), f"{container}:/tmp/model.patch"],
            log_path,
        )
        run_command(
            [
                "docker",
                "exec",
                "-w",
                workdir,
                container,
                "git",
                "-c",
                "safe.directory=*",
                "apply",
                "--binary",
                "/tmp/model.patch",
            ],
            log_path,
            timeout=300,
        )

    timeout = int(config["verifier"]["timeout_sec"])
    with (case_dir / "verifier.log").open("wb") as verifier_log:
        test_result = run_command(
            [
                "docker",
                "exec",
                "-w",
                workdir,
                container,
                "timeout",
                "--signal=KILL",
                str(timeout),
                "bash",
                "/tests/test.sh",
            ],
            log_path,
            timeout=timeout + 15,
            output=verifier_log,
            check=False,
        )
    if test_result.returncode not in (0, 1):
        raise RuntimeError(
            f"官方判分进程异常退出，退出码 {test_result.returncode}"
        )
    reward = (verifier_logs / "reward.txt").read_text(encoding="utf-8").strip()
    if reward not in ("0", "1"):
        raise ValueError(f"官方判分产生无效 reward：{reward!r}")
    if reward == "1" and test_result.returncode != 0:
        raise RuntimeError(
            f"官方 reward 与判分进程退出状态不一致："
            f"{reward=}, {test_result.returncode=}"
        )
    return {
        "resolved": reward == "1",
        "verifier_returncode": test_result.returncode,
    }
