"""把两段式判定组装成一个可注入 ChatService 的 guard 函数。

make_guard 是"编排"：先跑免费规则，没命中再花 token 请模型分类。
这段不是练习部分，是给你看两段是怎么接起来的。
"""

from __future__ import annotations

from collections.abc import Callable

from agent_lab.llm.client import LLMClient
from agent_lab.models import Message
from agent_lab.routing.classifier import IntentResult, classify_with_model
from agent_lab.routing.rules import match_out_of_scope_rule

# 分类模型看到的系统提示词：这是"内部协议"，要求它只输出 JSON。
# 返回的 JSON 结构由 classifier.py 里的 _INTENTS 约束。
_CLASSIFIER_PROMPT = (
    "你是电商客服的意图分类器。只判断用户问题是否属于订单、商品、售后范围。"
    '只输出 JSON，不要输出任何其他内容，格式：'
    '{"intent": "in_scope", "reason": "一句话说明判断依据"}'
    "（intent 只能是 in_scope 或 out_of_scope）"
)


def make_guard(
    client: LLMClient,
    *,
    shop_name: str,
    fallback: IntentResult | None = None,
) -> Callable[[str], IntentResult]:
    """生成一个越界判定函数：先规则、后模型。

    - 规则命中 -> 直接判越界，不花钱；
    - 规则没命中 -> 把问题交给分类模型；
    - 模型结果不可靠时，用 fallback 兜住（fail-closed，默认判越界）。
    """
    default_fallback = IntentResult(
        in_scope=False, intent="unknown", reason="分类结果不可靠，保守拒绝"
    )

    def guard(text: str) -> IntentResult:
        rule = match_out_of_scope_rule(text)
        if rule is not None:
            return IntentResult(
                in_scope=False, intent=f"rule:{rule.name}", reason=rule.reason
            )
        messages = [
            Message(
                message_id="system:router", session_id="", role="system",
                content=_CLASSIFIER_PROMPT,
            ),
            Message(
                message_id="user:router", session_id="", role="user", content=text
            ),
        ]
        return classify_with_model(client, messages, fallback=fallback or default_fallback)

    return guard
