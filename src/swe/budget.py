# 提供单题解题的命令行入口，保存任务、思考模式和预算，并在独立子进程中启动智能体。
# 父进程监管整体解题时限，结束或异常时回收子进程、停止任务容器。
# 汇总耗时，并从轨迹恢复轮数、词元（token）和费用统计，保存供独立评测流程读取的结果。

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from swe.cost import read_run_usage

# 整体解题时限为 50 分钟。
TASK_TIMEOUT_SECONDS = 3000


def _write_json(path: Path, data: dict) -> None:
    """先写同目录的临时文件，再替换目标文件，避免读取方看到半份 JSON。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


# 在工作子进程中读取运行参数、执行智能体，并保存智能体的结束摘要。
def _run_worker(run_dir: Path) -> None:
    result_path = run_dir / "agent_result.json"
    try:
        # 只有实际解题的子进程才加载模型依赖。
        from swe.mini_agent import run_agent

        case = json.loads((run_dir / "case.json").read_text(encoding="utf-8"))
        result = run_agent(
            case["task"],
            container_name=case["container"],
            workdir=case["workdir"],
            timeout=case["command_timeout"],
            max_steps=case["max_steps"],
            cost_limit=float(case["cost_limit_usd"]),
            trace_path=run_dir / "trajectory.jsonl",
            thinking_mode=case["thinking_mode"],
        )
    except Exception as error:
        try:
            _write_json(result_path, {
                "exit_reason": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        except OSError as write_error:
            error.add_note(f"保存工作进程错误失败：{write_error}")
        raise

    # 对话已经逐条保存在轨迹中，结果文件只保留摘要，避免重复写入大段历史。
    result.pop("history")
    _write_json(result_path, result)


def run_with_budget(task: str, container: str, workdir: str, run_dir: Path, *,
                    thinking_mode: str = "disabled") -> dict:
    """启动并监管一次解题，停止传入的容器并汇总运行统计。

    正常结束、超时或工作进程失败时，均从轨迹恢复已记录的用量。
    """

    def read_worker_result(
        path: Path,
        *,
        returncode: int | None,
        timed_out: bool,
        startup_error: OSError | None,
    ) -> dict:
        """根据工作进程的退出情况读取解题结果。"""
        if startup_error is not None:
            return {
                "exit_reason": "error",
                "error_type": type(startup_error).__name__,
                "error": str(startup_error),
            }
        if timed_out:
            return {"exit_reason": "time_limit"}

        result = {}
        if returncode == 0 or path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
        if returncode != 0:
            result.update(exit_reason="error", worker_returncode=returncode)
        return result

    # 每次使用独立目录，禁止覆盖已经开始过的尝试。
    run_dir.mkdir(parents=True, exist_ok=False)
    # 固定本次任务的输入和预算，供工作进程读取并留作后续检查。
    _write_json(run_dir / "case.json", {
        "task": task,
        "container": container,
        "workdir": workdir,
        "max_steps": 250,
        "command_timeout": 60,
        "cost_limit_usd": "3",
        "task_timeout_seconds": TASK_TIMEOUT_SECONDS,
        "thinking_mode": thinking_mode,
    })

    # 初始化结束状态，并用单调时钟计时，避免系统时间调整影响时限。
    process = None
    timed_out = False
    startup_error = None
    cleanup_error = None
    started = time.monotonic()

    # 将智能体放入独立 Python 进程，并把标准输出和标准错误合并保存到日志。
    try:
        with (run_dir / "agent.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable, "-B", "-u", "-m",
                    "swe.budget",
                    "--worker", str(run_dir),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

            # 扣除进程创建耗时后再等待，保证启动开销也计入整体解题时限。
            remaining = max(
                0, TASK_TIMEOUT_SECONDS - (time.monotonic() - started)
            )
            process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out = True
    except OSError as error:
        if process is not None:
            raise
        startup_error = error
    finally:
        # 工作进程仍在运行时强制结束并等待回收，阻止其继续请求模型。
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

        # 解题耗时不包含随后停止容器的时间。
        elapsed_seconds = round(time.monotonic() - started, 3)

        # 容器由本题使用，即使未能启动工作进程也需要停止。
        try:
            subprocess.run(
                ["docker", "stop", "--time", "0", container],
                capture_output=True,
                check=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # 暂存清理错误，先完成统计落盘，再向调用方报告失败。
            cleanup_error = f"{type(error).__name__}: {error}"

    result_path = run_dir / "agent_result.json"
    result = read_worker_result(
        result_path,
        returncode=process.returncode if process is not None else None,
        timed_out=timed_out,
        startup_error=startup_error,
    )

    # 从轨迹恢复已知消耗，避免超时或异常覆盖此前已经产生的费用和轮数。
    # 最后补充解题耗时及容器清理状态，写出供调用方读取的结果。
    result.update(read_run_usage(run_dir / "trajectory.jsonl"))
    result["seconds"] = elapsed_seconds
    # 即使工作进程超时或失败，结果也保留本次实际采用的思考模式。
    result["thinking_mode"] = thinking_mode
    result["cleanup_error"] = cleanup_error
    _write_json(result_path, result)

    # 落盘后再报告错误，启动失败时保留原始异常。
    if startup_error is not None:
        raise startup_error
    if cleanup_error is not None:
        raise RuntimeError(f"本题统计已保存，但容器停止失败：{cleanup_error}")
    return result


# 解析命令行参数，选择内部工作进程模式或对外提供的预算监管入口。
def main() -> None:
    # 定义普通调用所需参数；隐藏的 --worker 参数只用于父进程启动子进程。
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--container")
    parser.add_argument("--workdir", default="/app")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--thinking-mode", choices=("disabled", "max"), default="disabled"
    )
    args = parser.parse_args()

    # 内部工作进程直接读取已有目录开始解题，避免再次启动一层监管进程。
    if args.worker is not None:
        _run_worker(args.worker)
        return

    # 普通调用必须提供任务文件、已有容器和输出目录，缺失时由参数解析器报错。
    if args.task_file is None or args.container is None or args.output is None:
        parser.error("必须提供 --task-file、--container 和 --output")

    # 读取任务文本，并使用绝对输出路径启动带预算的完整运行流程。
    result = run_with_budget(
        args.task_file.read_text(encoding="utf-8"),
        args.container,
        args.workdir,
        args.output.resolve(),
        thinking_mode=args.thinking_mode,
    )
    # 将统计打印为 JSON，便于终端查看或由外部调度程序接收。
    print(json.dumps(result, ensure_ascii=False))

    # 工作进程失败时返回非零退出码；预算用尽等结束原因由结果字段说明。
    if result["exit_reason"] == "error":
        raise SystemExit(1)


# 从命令行直接运行本模块时启用参数解析和预算入口。
if __name__ == "__main__":
    main()
