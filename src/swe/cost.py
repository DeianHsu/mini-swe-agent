# 依据模型响应中的词元（token）用量和固定价目，分别计算缓存命中输入、未命中输入及输出的估算费用。
# 从轨迹恢复请求轮数、用量和费用；遇到缺失响应或未写完整的记录时，保留已知消耗并标记统计不完整。

import argparse
import json
from decimal import Decimal
from pathlib import Path

# 用于估算 deepseek-flash 费用的固定高峰价，单位为美元/百万词元。
TOKEN_PRICES_PER_MILLION = {
    "prompt_cache_hit_tokens": Decimal("0.006"),
    "prompt_cache_miss_tokens": Decimal("0.30"),
    "completion_tokens": Decimal("1.20"),
}
TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


def summarize_reply_usage(
    usage: dict | None,
    prices: dict[str, Decimal] = TOKEN_PRICES_PER_MILLION,
) -> tuple[dict[str, int], Decimal]:
    """核对单轮用量，返回词元数和美元费用。"""
    if usage is None:
        raise ValueError("模型响应缺少 usage，无法核算费用")
    tokens = {}
    for field in TOKEN_FIELDS:
        count = usage[field]
        if type(count) is not int or count < 0:
            raise ValueError(
                f"用量字段 {field} 必须是非负整数，收到 {count!r}"
            )
        tokens[field] = count
    cost = sum(
        tokens[field] * price for field, price in prices.items()
    ) / Decimal("1000000")
    return tokens, cost


def read_run_usage(trace_path: Path) -> dict:
    """从运行轨迹中恢复已确认的用量和成本。"""
    tokens = dict.fromkeys(TOKEN_FIELDS, 0)
    model_calls = 0
    accounted_calls = 0
    known_cost = Decimal("0")
    # 轨迹不存在时统计视为不完整，价目必须从本次运行的起始事件中读取。
    usage_complete = trace_path.exists()
    prices = None

    if usage_complete:
        # 使用二进制逐行读取，避免强制终止留下的半个 UTF-8 字符导致解码失败。
        with trace_path.open("rb") as handle:
            for line in handle:
                # 缺少结尾换行表示事件尚未完整写入，保留此前统计并停止读取。
                if not line.endswith(b"\n"):
                    usage_complete = False
                    break

                # 解析单条事件，再按事件类型更新价目、请求数或响应消耗。
                event = json.loads(line)
                data = event["data"]

                if event["kind"] == "start":
                    # 沿用运行时保存的价目，避免后来修改价格影响历史费用。
                    prices = {
                        key: Decimal(value)
                        for key, value in data[
                            "token_prices_per_million"
                        ].items()
                    }

                elif event["kind"] == "request":
                    # 每条 request 事件计一次请求尝试，事件在实际发送前记录。
                    model_calls += 1

                elif event["kind"] == "assistant":
                    # 使用和实时预算相同的单轮用量规则。
                    if prices is None:
                        raise ValueError("轨迹缺少起始价目，无法计算费用")
                    try:
                        reply_tokens, cost = summarize_reply_usage(
                            data["usage"], prices
                        )
                    except (KeyError, TypeError, ValueError):
                        # 此响应无法核算，保留其他响应的已知成本。
                        usage_complete = False
                        continue

                    for field, count in reply_tokens.items():
                        tokens[field] += count
                    known_cost += cost
                    accounted_calls += 1

    # 用量完整要求起始价目存在，已记录的事件均写完整，模型回复均可核算。
    # 请求尝试数与可核算的回复数也必须一致。
    usage_complete = (
        usage_complete
        and prices is not None
        and model_calls == accounted_calls
    )

    # 同时返回已知成本和完整性标志；统计不完整时不把部分成本当作完整费用。
    # 总词元仅计算输入加输出，避免重复计算缓存命中的输入。
    return {
        "model_calls": model_calls,
        "accounted_calls": accounted_calls,
        **tokens,
        "total_tokens": tokens["prompt_tokens"] + tokens["completion_tokens"],
        "known_cost_usd": str(known_cost),
        "budget_cost_usd": str(known_cost) if usage_complete else None,
        "usage_complete": usage_complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="读取运行轨迹并输出用量汇总 JSON")
    parser.add_argument("trace", type=Path, help="运行目录中的 trajectory.jsonl")
    args = parser.parse_args()
    print(json.dumps(read_run_usage(args.trace), ensure_ascii=False))


if __name__ == "__main__":
    main()
