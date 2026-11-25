# 面向本地 Git 仓库的一键沙箱排错入口：
# 在断网且限额的临时 Docker 容器中拷贝代码副本，运行智能体定位并修复问题，
# 结束后导出标准 git diff 补丁与运行摘要，并在 finally 中强制清理临时容器。

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from textwrap import dedent
from uuid import uuid4

from swe import budget

DEFAULT_IMAGE = "python:3.14-slim"
CONTAINER_WORKDIR = "/workspace"


def build_fix_task(issue: str) -> str:
    """将报错信息或问题说明封装为适合自主修复的任务提示。"""
    cleaned = issue.strip()
    if not cleaned:
        raise ValueError("问题描述不能为空")
    return dedent(f"""
        请排查并修复当前 Git 仓库中的问题：

        {cleaned}

        要求：
        1. 先阅读相关源码和测试，确认根因后再做最小必要修改。
        2. 修改后运行相关测试验证修复生效，不要提交新的 Git commit。
        3. 确认修复并通过验证后，简述修改内容并以普通文本回复 finish。
    """).strip()


def _run_docker(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    """执行宿主机 Docker 命令并在失败时附带错误信息。"""
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=timeout,
    )


def _export_patch(container_name: str, workdir: str) -> str:
    """短暂拉起已停止的任务容器并导出含未跟踪新文件的标准 Git 补丁。"""
    _run_docker(["start", container_name])
    try:
        script = (
            "git config --global --add safe.directory \"$PWD\" && "
            "git add -N . && "
            "git diff HEAD --"
        )
        completed = _run_docker(
            ["exec", "-w", workdir, container_name, "bash", "-c", script]
        )
        return completed.stdout
    finally:
        subprocess.run(
            ["docker", "stop", "--time", "0", container_name],
            capture_output=True,
            check=False,
            timeout=60,
        )


def fix_local_repo(
    repo_dir: Path,
    issue: str,
    output_dir: Path,
    *,
    image: str = DEFAULT_IMAGE,
    thinking_mode: str = "disabled",
    cpus: str = "1",
    memory: str = "4g",
) -> dict:
    """在隔离 Docker 沙箱中修复本地 Git 仓库并导出 fix.patch 与摘要。"""
    repo_path = repo_dir.resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"仓库目录不存在：{repo_path}")
    if not (repo_path / ".git").exists():
        raise ValueError(f"目录不是 Git 仓库（缺少 .git）：{repo_path}")

    task = build_fix_task(issue)
    output_path = output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    run_dir = output_path / "run"
    container_name = f"mini-swe-fix-{uuid4().hex[:12]}"
    container_created = False

    try:
        # 使用断网和资源限额容器，并通过 docker cp 拷贝代码副本，避免修改宿主机工作区。
        _run_docker([
            "run", "-d",
            "--name", container_name,
            "--network", "none",
            "--cpus", str(cpus),
            "--memory", str(memory),
            "-w", CONTAINER_WORKDIR,
            image,
            "sleep", "infinity",
        ])
        container_created = True
        _run_docker(["cp", f"{repo_path}/.", f"{container_name}:{CONTAINER_WORKDIR}"])

        agent_result = budget.run_with_budget(
            task,
            container_name,
            CONTAINER_WORKDIR,
            run_dir,
            thinking_mode=thinking_mode,
        )

        patch_text = _export_patch(container_name, CONTAINER_WORKDIR)
        patch_path = output_path / "fix.patch"
        patch_path.write_text(patch_text, encoding="utf-8", newline="\n")

        summary = {
            "repo": str(repo_path),
            "image": image,
            "exit_reason": agent_result.get("exit_reason"),
            "model_calls": agent_result.get("model_calls"),
            "transient_retries": agent_result.get("transient_retries", 0),
            "known_cost_usd": agent_result.get("known_cost_usd"),
            "seconds": agent_result.get("seconds"),
            "patch_path": str(patch_path),
            "patch_bytes": len(patch_text.encode("utf-8")),
            "run_dir": str(run_dir),
        }
        (output_path / "fix_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        if container_created:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                check=False,
                timeout=60,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在断网 Docker 沙箱中自动定位并修复本地 Git 仓库 Bug，导出 fix.patch"
    )
    parser.add_argument("--repo", type=Path, required=True, help="本地 Git 仓库目录")
    issue_group = parser.add_mutually_exclusive_group(required=True)
    issue_group.add_argument("--issue", help="直接传入报错日志或问题描述")
    issue_group.add_argument(
        "--issue-file", type=Path, help="包含问题描述的 UTF-8 文本文件"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录（默认写入 archive/fix/<时间戳>）",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="沙箱 Docker 镜像")
    parser.add_argument(
        "--thinking-mode", choices=("disabled", "max"), default="disabled"
    )
    parser.add_argument("--cpus", default="1", help="容器 CPU 上限")
    parser.add_argument("--memory", default="4g", help="容器内存上限")
    args = parser.parse_args()

    issue_text = (
        args.issue_file.read_text(encoding="utf-8")
        if args.issue_file is not None
        else args.issue
    )
    output_dir = args.output or (
        Path("archive/fix") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    summary = fix_local_repo(
        args.repo,
        issue_text,
        output_dir,
        image=args.image,
        thinking_mode=args.thinking_mode,
        cpus=args.cpus,
        memory=args.memory,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["exit_reason"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
