# 依据模型响应中的词元（token）用量和固定价目，分别计算缓存命中输入、未命中输入及输出的估算费用。
# 从轨迹恢复请求轮数、用量和费用；遇到缺失响应或未写完整的记录时，保留已知消耗并标记统计不完整。

import json
from decimal import Decimal
from pathlib import Path

# 以固定高峰价估算费用，单位为美元/百万词元（token）；本表适用于 deepseek-flash。
TOKEN_PRICES = {
    "prompt_cache_hit_tokens": Decimal("0.006"),
    "prompt_cache_miss_tokens": Decimal("0.30"),
    "completion_tokens": Decimal("1.20"),
}


# 分别计算缓存命中输入、未命中输入和输出的费用，返回美元金额。
def calculate_cost(usage: dict, prices: dict = TOKEN_PRICES) -> Decimal:
    # 按每百万词元的价格折算实际消耗，并使用 Decimal 保留金额精度。
    return sum(
        usage[field] * price for field, price in prices.items()
    ) / Decimal("1000000")


# 逐条读取完整轨迹，恢复请求轮数、词元用量和成本，并标记统计是否完整。
def read_usage(path: Path) -> dict:
    # 从零累计已确认消耗，并分别记录已发起的请求数和可核算的响应数。
    tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    model_calls = 0
    accounted_calls = 0
    known_cost = Decimal("0")
    # 轨迹不存在时统计视为不完整，价目必须从本次运行的起始事件中读取。
    complete = path.exists()
    prices = None

    if complete:
        # 使用二进制逐行读取，避免强制终止留下的半个 UTF-8 字符导致解码失败。
        with path.open("rb") as handle:
            for line in handle:
                # 缺少结尾换行表示事件尚未完整写入，保留此前统计并停止读取。
                if not line.endswith(b"\n"):
                    complete = False
                    break

                # 解析单条事件，再按事件类型更新价目、请求数或响应消耗。
                event = json.loads(line)
                data = event["data"]

                if event["kind"] == "start":
                    # 沿用运行时保存的价目，避免后来修改价格影响历史费用。
                    prices = {
                        key: Decimal(value)
                        for key, value in data["token_prices_per_million"].items()
                    }

                elif event["kind"] == "request":
                    # 请求发起后即计入轮数，后续调用失败也不会丢掉这一轮记录。
                    model_calls += 1

                elif event["kind"] == "assistant":
                    # 仅用模型返回的 usage 记账，缺失时保留其他响应的已知消耗。
                    usage = data["usage"]
                    if usage is None:
                        complete = False
                        continue

                    # 没有起始价目就无法核算费用，直接报告轨迹数据错误。
                    if prices is None:
                        raise ValueError("轨迹缺少起始价目，无法计算费用")

                    # 本轮各项用量及费用都可计算时，才更新累计统计。
                    try:
                        updated_tokens = {key: total + usage[key] for key, total in tokens.items()}
                        cost = calculate_cost(usage, prices)
                    except (KeyError, TypeError):
                        # 此响应无法核算，保留其他响应的已知成本。
                        complete = False
                        continue

                    # 缓存命中量作为输入的细分项保存，不另外计入总词元数。
                    tokens = updated_tokens
                    known_cost += cost
                    accounted_calls += 1

    # 只有轨迹完整、价目已知且每个请求都能核算费用时，统计才算完整。
    complete = (
        complete
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
        "budget_cost_usd": str(known_cost) if complete else None,
        "usage_complete": complete,
    }
