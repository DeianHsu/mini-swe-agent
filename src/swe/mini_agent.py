# 实现智能体的解题循环：组装提示和历史，调用模型、执行工具，并将工具结果反馈给下一轮。
# 管理轮数和费用阈值、裁剪工具历史，将对话、用量及结束状态写入指定的运行轨迹。
# 整体任务时限与子进程监管由 budget 模块负责，本模块直接运行时提供只读的容器演示任务。

import json
from decimal import Decimal
from pathlib import Path
from textwrap import dedent

from swe.cost import TOKEN_PRICES_PER_MILLION, summarize_reply_usage
from swe.environment import execute, limit_output
from swe.use_llm import create_client, send_messages


# 为历史工具输出设置总字符上限，并用占位文本替换过早的大段输出。
MAX_TOOL_HISTORY_CHARS = 256000
OMITTED_OUTPUT = "[更早的命令输出已省略。需要时请缩小范围重新读取。]"


# 原地压缩历史工具输出，保留消息顺序与工具调用标识之间的对应关系。
def trim_tool_history(history: list[dict]) -> None:
    # 只处理含文本的工具消息，先限制每条输出的长度。
    tool_messages = [
        message for message in history
        if message.get("role") == "tool" and isinstance(message.get("content"), str)
    ]
    for message in tool_messages:
        message["content"] = limit_output(message["content"])
    # 若总量仍超限，从最早的工具输出开始替换，优先保留近期信息。
    total = sum(len(message["content"]) for message in tool_messages)
    for message in tool_messages:
        if total <= MAX_TOOL_HISTORY_CHARS:
            break
        # 只替换比占位文本更长的内容，避免压缩后反而增加字符数。
        if len(message["content"]) > len(OMITTED_OUTPUT):
            total -= len(message["content"]) - len(OMITTED_OUTPUT)
            message["content"] = OMITTED_OUTPUT


def ask_model(client, task: str, history: list[dict], workdir: str, *,
              thinking_mode: str) -> tuple[dict, dict]:
    """组装任务和历史消息并请求一轮模型回复。"""
    prompt = dedent(f"""
        你是一个编程智能体。
        执行环境是 Linux 容器中的 Bash，当前目录为 {workdir}。
        需要操作时，通过 execute 工具提交 Bash 脚本。
        每次调用启动独立进程，文件修改会保留。
        任务完成后，以普通文本回复 finish。
    """).strip()
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": task}, *history]
    return send_messages(client, messages, thinking_mode=thinking_mode)


# 将运行事件按一行一条 JSON 的格式追加到轨迹文件，供后续恢复用量和排查问题。
def record_event(path: Path | None, kind: str, data: dict) -> None:
    # 未指定路径时不落盘；指定路径时每条事件写完即关闭文件，减少中断时的记录丢失。
    if path is not None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {"kind": kind, "data": data},
                    ensure_ascii=False,
                ) + "\n"
            )


def _run_tool(tool_call: dict, *, container_name: str, workdir: str, timeout: int) -> str:
    """解析单次工具调用，将格式错误作为工具结果返回。"""
    function = tool_call["function"]
    if function["name"] != "execute":
        return f"Error: 未知工具 {function['name']}"
    try:
        arguments = json.loads(function["arguments"])
    except (json.JSONDecodeError, TypeError) as error:
        return f"Error: 参数解析失败：{error}"
    if not isinstance(arguments, dict):
        return "Error: 参数必须是 JSON 对象"
    command = arguments.get("command")
    if not isinstance(command, str):
        return "Error: command 必须是字符串"
    return execute(command, container_name=container_name, workdir=workdir, timeout=timeout)


def run_agent(task: str, *, container_name: str = "mini-swe-learning",
              workdir: str = "/workspace", timeout: int = 60,
              max_steps: int = 250, cost_limit: float = 3.0,
              trace_path: Path | None = None, thinking_mode: str = "disabled") -> dict:
    """循环请求模型并执行工具，直到模型结束或用完轮数、费用预算。"""
    if max_steps < 1 or timeout <= 0 or cost_limit <= 0:
        raise ValueError("轮数、命令时限和费用阈值必须大于零")

    history = []
    total_cost = Decimal("0")
    cost_known = True
    cost_limit_usd = Decimal(str(cost_limit))
    model_calls = 0
    exit_reason = "step_limit"

    # 保存本次任务、预算及实际采用的价目，使后续统计沿用本次运行的计价口径。
    record_event(trace_path, "start", {
        "task": task,
        "workdir": workdir,
        "max_steps": max_steps,
        "command_timeout": timeout,
        "cost_limit_usd": str(cost_limit_usd),
        "thinking_mode": thinking_mode,
        "token_prices_per_million": {
            key: str(value) for key, value in TOKEN_PRICES_PER_MILLION.items()
        },
    })

    try:
        with create_client() as client:
            for _ in range(max_steps):
                # 在下一轮请求前检查费用。刚返回的一轮可能使累计费用超过阈值。
                if total_cost >= cost_limit_usd:
                    exit_reason = "cost_limit"
                    break

                # 发出请求前先将本轮费用标为未知，避免中断时把缺失消耗当作零。
                model_calls += 1
                cost_known = False
                record_event(trace_path, "request", {"step": model_calls})
                trim_tool_history(history)
                message, metadata = ask_model(client, task, history, workdir, thinking_mode=thinking_mode)
                history.append(message)

                # 工具执行前先保存回复与用量，以便在命令失败后恢复费用。
                record_event(trace_path, "assistant", {"step": model_calls, "message": message, **metadata})
                _, reply_cost = summarize_reply_usage(metadata["usage"])
                total_cost += reply_cost
                cost_known = True

                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    exit_reason = "finished" if metadata["finish_reason"] == "stop" else "model_incomplete"
                    break

                for tool_call in tool_calls:
                    tool_output = _run_tool(tool_call, container_name=container_name, workdir=workdir, timeout=timeout)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_output,
                    }
                    history.append(tool_message)
                    record_event(trace_path, "tool", tool_message)

    except Exception as error:
        # 记录本层无法恢复的异常，再向调用方抛出，避免把执行失败伪装成正常结束。
        exit_reason = "error"
        record_event(trace_path, "error", {
            "type": type(error).__name__,
            "message": str(error),
        })
        raise

    finally:
        # 保存结束原因、轮数和已确认费用；最后一次请求的用量未知时，费用保持 None。
        summary = {
            "exit_reason": exit_reason,
            "model_calls": model_calls,
            "thinking_mode": thinking_mode,
            "budget_cost_usd": str(total_cost) if cost_known else None,
        }
        record_event(trace_path, "end", summary)

    # 将结束摘要与对话历史交给调用方，由调用方决定展示或保存哪些内容。
    return {**summary, "history": history}


# 直接运行此模块时，通过预算入口执行一个只读的容器演示任务。
if __name__ == "__main__":
    # 演示所需的监管入口仅在直接运行时加载，导入智能体函数时不加载它。
    from uuid import uuid4

    from swe.budget import run_with_budget

    # 演示任务用于检查工作目录与文件读取是否可用。
    task = dedent(r"""
        查看当前工作目录，列出其中的文件，读取 hello.txt 的完整内容。
        只执行查看操作，完成后回复 finish。
    """).strip()

    # 为每次演示分配唯一输出目录，并启动带整体时限的解题进程。
    run_dir = Path("archive/demo") / uuid4().hex
    result = run_with_budget(
        task,
        "mini-swe-learning",
        "/workspace",
        run_dir,
    )
    # 展示轨迹保存位置和最终摘要，方便查看本次运行结果。
    print(f"轨迹目录：{run_dir.resolve()}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["exit_reason"] == "error":
        raise SystemExit(1)
