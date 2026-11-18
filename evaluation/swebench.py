"""SWE-bench Pro V2 独立评测：只通过命令行和 JSON 产物调用主线。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from uuid import uuid4

import verifier
from verifier import DATASET_REPOSITORY, DATASET_REVISION

TASK_COUNT = 642
BATCH_SIZE = 214

# 使用原始镜像的实际文件作为基准，包含构建时留下的未提交、未跟踪文件。
# 只读工作区由原始镜像中的 Git 读取，不信任智能体修改过的 .git。
CAPTURE_SCRIPT = r"""
set -euo pipefail
repo="$1"
candidate="$2"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
git -c safe.directory='*' -c core.hooksPath=/dev/null -c core.fsmonitor=false \
    --git-dir="$repo/.git" --work-tree="$repo" add -A
baseline=$(git -c safe.directory='*' --git-dir="$repo/.git" write-tree)
git -c safe.directory='*' -c core.hooksPath=/dev/null -c core.fsmonitor=false \
    --git-dir="$repo/.git" --work-tree="$candidate" add -A
git -c safe.directory='*' --git-dir="$repo/.git" --work-tree="$candidate" \
    diff --cached --binary --no-ext-diff --no-textconv "$baseline" --
"""


def make_manifest(dataset: Path, run_id: str, code_revision: str, *,
                  thinking_mode: str = "disabled") -> dict:
    config_paths = sorted((dataset / "v2/tasks").glob("*/task.toml"))
    if len(config_paths) != TASK_COUNT:
        raise ValueError(f"全量评测需要 {TASK_COUNT} 题，实际找到 {len(config_paths)} 题")

    instances = []
    for config_path in config_paths:
        task_dir = config_path.parent
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if config["agent"]["network_mode"] != "no-network" or config["agent"]["timeout_sec"] != 3000:
            raise ValueError(f"{task_dir.name}: 与锁定的断网、50 分钟协议不符")
        for filename in ("instruction.md", "tests/test.sh", "tests/config.json"):
            if not (task_dir / filename).is_file():
                raise FileNotFoundError(task_dir / filename)
        instances.append(task_dir.name)

    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "thinking_mode": thinking_mode,
        "instances": instances,
        "budget": {
            "max_steps": 250,
            "command_timeout": 60,
            "cost_limit_usd": "3",
            "task_timeout_seconds": 3000,
        },
        "attempts_per_task": 1,
        "agent_network": "none",
        "fresh_verifier": True,
    }


def run_budget_cli(task_file: Path, container: str, workdir: str, run_dir: Path,
                   log_path: Path, *, thinking_mode: str = "disabled") -> int:
    # 独立进程组用于在主线监管进程异常卡住时一并终止其子进程。
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", "-u", "-m", "swe.budget",
             "--task-file", str(task_file), "--container", container,
             "--workdir", workdir, "--output", str(run_dir), "--thinking-mode", thinking_mode],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            # 主线执行 3000 秒预算，额外时间只供停止容器和保存统计。
            return process.wait(timeout=3180)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return process.returncode


def _read_agent_result(run_dir: Path, log_path: Path) -> dict:
    result_path = run_dir / "agent_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    if "usage_complete" in result:
        return result

    # 监管进程提前退出时，轨迹可能已经有用量，但汇总结果还没写出。
    trace_path = run_dir / "trajectory.jsonl"
    if trace_path.is_file():
        try:
            read = verifier.run_command([sys.executable, "-B", "-m", "swe.cost", str(trace_path)], log_path)
            result.update(json.loads(read.stdout))
            return result
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            result["usage_error"] = str(error)

    # 只有工作进程摘要时，仍可保留摘要中的已知费用。
    if result.get("budget_cost_usd") is not None:
        result.update(known_cost_usd=result["budget_cost_usd"], budget_cost_usd=None)
    return result


def _export_patch(case_dir: Path, image_id: str, workdir: str, limits: list[str],
                  volume_name: str, capture_name: str) -> Path:
    patch_path = case_dir / "model.patch"
    with patch_path.open("wb") as patch:
        verifier.run_command(
            ["docker", "run", "--rm", "--name", capture_name, "--network", "none",
             *limits, "--mount", f"type=volume,source={volume_name},target=/candidate,readonly",
             "--entrypoint", "bash", image_id, "-c", CAPTURE_SCRIPT, "capture", workdir, "/candidate"],
            case_dir / "runner.log", timeout=300, output=patch,
        )
    return patch_path


def eval_case(dataset: Path, manifest: dict, index: int, output: Path) -> dict:
    if not 0 <= index < len(manifest["instances"]):
        raise ValueError("题目序号超出清单范围")
    instance_id = manifest["instances"][index]
    task_dir = dataset / "v2/tasks" / instance_id
    case_dir = output.resolve() / instance_id
    case_dir.mkdir(parents=True, exist_ok=False)
    verifier.save_json(case_dir / "case.json", {"instance_id": instance_id, "run_id": manifest["run_id"]})

    result = {
        "instance_id": instance_id,
        "run_id": manifest["run_id"],
        "thinking_mode": manifest["thinking_mode"],
        "status": "running",
        "stage": "setup",
        "resolved": None,
        "seconds": 0.0,
        "model_calls": 0,
        "accounted_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "known_cost_usd": "0",
        "budget_cost_usd": "0",
        "usage_complete": True,
    }
    result_path = case_dir / "agent_result.json"
    verifier.save_json(result_path, result)
    log_path = case_dir / "runner.log"
    prefix = "swebench-" + uuid4().hex[:16]
    agent_name, capture_name, verifier_name = (prefix + suffix for suffix in ("-agent", "-capture", "-verify"))
    volume_name = prefix + "-worktree"
    containers = []
    volume_created = False
    started = time.monotonic()

    try:
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        limits = verifier.container_limits(config)
        task_file = case_dir / "instruction.md"
        task_file.write_text((task_dir / "instruction.md").read_text(encoding="utf-8"), encoding="utf-8")

        containers.append(capture_name)
        image_id, workdir, base_commit = verifier.prepare_image(config, capture_name, log_path)
        result["image_id"] = image_id
        result["base_commit"] = base_commit

        verifier.run_command(["docker", "volume", "create", volume_name], log_path)
        volume_created = True
        containers.append(agent_name)
        # 空卷由镜像中的仓库初始化；不挂载宿主机、测试目录或任何密钥。
        verifier.run_command(
            ["docker", "run", "-d", "--name", agent_name, "--network", "none",
             *limits, "--env", "SDL_AUDIODRIVER=dummy",
             "--mount", f"type=volume,source={volume_name},target={workdir}",
             "--entrypoint", "sleep", image_id, "infinity"], log_path,
            timeout=config["environment"]["build_timeout_sec"],
        )

        result.update(stage="agent", seconds=None, model_calls=None, accounted_calls=None,
                      prompt_tokens=None, completion_tokens=None, total_tokens=None,
                      budget_cost_usd=None, usage_complete=False)
        verifier.save_json(result_path, result)
        agent_dir = case_dir / "agent"
        result["agent_returncode"] = run_budget_cli(
            task_file, agent_name, workdir, agent_dir, case_dir / "budget.log",
            thinking_mode=manifest["thinking_mode"],
        )
        result.update(_read_agent_result(agent_dir, log_path))
        # 预算记录必须存在且匹配；缺失时保留消耗，但不能发布有效判分。
        settings = json.loads((agent_dir / "case.json").read_text(encoding="utf-8"))
        result["budget"] = {key: settings[key] for key in manifest["budget"]}
        if result["budget"] != manifest["budget"]:
            raise ValueError("主线实际预算与本轮预定预算不一致")
        # 模式与预算一样必须来自本轮清单，避免把关闭思考的结果混入最高档评测。
        if settings["thinking_mode"] != manifest["thinking_mode"]:
            raise ValueError("主线实际思考模式与本轮预定模式不一致")

        result["stage"] = "capture"
        verifier.save_json(result_path, result)
        # 不恢复智能体容器；冻结工作区后交给干净镜像中的 Git 导出补丁。
        verifier.run_command(["docker", "stop", "--time", "0", agent_name], log_path)
        patch_path = _export_patch(case_dir, image_id, workdir, limits, volume_name, capture_name)

        result["stage"] = "verifier"
        verifier.save_json(result_path, result)
        containers.append(verifier_name)
        result.update(verifier.verify_patch(task_dir, config, image_id, workdir,
                                            patch_path, case_dir, verifier_name))
        result.update(status="graded", stage="done")

    except Exception as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
    finally:
        cleanup_errors = []
        for name in reversed(containers):
            try:
                # --rm 的探测/导出容器通常已自行删除；保留清理命令的日志。
                verifier.run_command(["docker", "rm", "-f", name], log_path, check=name != capture_name)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))
        if volume_created:
            try:
                verifier.run_command(["docker", "volume", "rm", volume_name], log_path)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))
        result["cleanup_errors"] = cleanup_errors
        result["evaluation_seconds"] = round(time.monotonic() - started, 3)
        verifier.save_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("revision")
    plan = commands.add_parser("plan")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--code-revision", required=True)
    plan.add_argument("--thinking-mode", choices=("disabled", "max"), default="disabled")
    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "revision":
        print(f"revision={DATASET_REVISION}")
    elif args.command == "plan":
        manifest = make_manifest(args.dataset, args.run_id, args.code_revision,
                                 thinking_mode=args.thinking_mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        verifier.save_json(args.output, manifest)
        for batch in range(3):
            indices = list(range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE))
            print(f"batch_{batch + 1}=" + json.dumps({"index": indices}, separators=(",", ":")))
    else:
        if sys.platform != "linux":
            parser.error("实际评测需在运行 Linux 容器的 Linux 主机上执行")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = eval_case(args.dataset, manifest, args.index, args.output)
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "graded" or result["cleanup_errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
