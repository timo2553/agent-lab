"""token 计量与成本估算。

两条原则：
- 价目表是"会过期"的数据，单独放一张表方便改；模型不在表里就返回 None，
  而不是瞎猜一个价格——客服系统里错的成本数字比没有数字更糟。
- 成本只用于观测和熔断，不作为计费依据。
"""

from __future__ import annotations

from agent_lab.llm.types import Usage
from agent_lab.models import UsageRecord

# 单位：元 / 百万 token，(输入价, 输出价)。示例价，以厂商官网为准。
PRICES: dict[str, tuple[float, float]] = {
    "deepseek-chat": (2.0, 8.0),
}


class BudgetExceededError(RuntimeError):
    """会话累计用量触到预算上限：在调用之前拒绝，而不是先花钱再报错。"""


def estimate_cost(
    model: str,
    usage: Usage,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """估算一次调用的费用；模型不在价目表里返回 None。"""
    table = PRICES if prices is None else prices
    price = table.get(model)
    if price is None:
        return None
    input_price, output_price = price
    return (
        usage.prompt_tokens / 1_000_000 * input_price
        + usage.completion_tokens / 1_000_000 * output_price
    )


def summarize(
    records: list[UsageRecord],
    prices: dict[str, tuple[float, float]] | None = None,
) -> tuple[Usage, float | None]:
    """把多条用量记录合成"累计用量 + 估算费用"。

    只要有一条记录的费用算不出来，整体费用就是 None（宁可说不确定，不给错数）。
    """
    total = Usage()
    cost: float | None = 0.0
    for record in records:
        total = total + record.usage
        one = estimate_cost(record.model, record.usage, prices)
        if one is None:
            cost = None
        elif cost is not None:
            cost += one
    return total, cost


def format_usage(usage: Usage, cost: float | None) -> str:
    text = (
        f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
        f"total={usage.total_tokens}"
    )
    if cost is None:
        return text + " | 费用未知（模型不在价目表里）"
    return text + f" | 约 ¥{cost:.4f}"