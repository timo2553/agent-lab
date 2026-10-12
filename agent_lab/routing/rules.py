"""规则前置：不花钱、不延迟的第一道闸门。

两段式越界判定的第一段。命中关键词直接判越界，根本不用调模型——
客服要挡的是"写代码、翻译、创作、其他领域咨询"，这些有明显的关键词特征。

注意定位：规则是"宁可放过、不可误伤"，还是反过来，取决于放行的代价。
这里只挡最典型的场景，拿不准的留给第二段模型分类（classifier.py）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """一条硬规则：命中任意 pattern（子串匹配）即判越界。"""

    name: str
    patterns: tuple[str, ...]
    reason: str


# 关键词按"业务语境下几乎不可能出现"来选，避免把正常订单咨询误伤。
# 这是会迭代的数据：上线后根据误报、漏报调整，规则层不参与业务逻辑。
OUT_OF_SCOPE_RULES: tuple[Rule, ...] = (
    Rule(
        name="coding",
        patterns=(
            "写代码", "写个程序", "写一段", "帮我写",
            "python", "javascript", "sql", "正则",
        ),
        reason="编程请求，超出客服范围",
    ),
    Rule(
        name="translation",
        patterns=("翻译", "translate"),
        reason="翻译请求，超出客服范围",
    ),
    Rule(
        name="content_creation",
        patterns=("写一篇", "写首诗", "写个故事", "写文案", "写作文", "写小说"),
        reason="内容创作请求，超出客服范围",
    ),
    Rule(
        name="other_domain",
        patterns=("数学题", "物理题", "医学建议", "法律建议", "怎么做饭"),
        reason="其他领域咨询，超出客服范围",
    ),
)


def match_out_of_scope_rule(text: str) -> Rule | None:
    """命中任意规则返回该规则，否则返回 None。"""
    for rule in OUT_OF_SCOPE_RULES:
        for pattern in rule.patterns:
            if pattern in text:
                return rule
    return None
