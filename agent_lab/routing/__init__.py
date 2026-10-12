"""越界判定：规则前置 + 模型分类（两段式）。"""

from agent_lab.routing.classifier import IntentResult, classify_with_model
from agent_lab.routing.guard import make_guard
from agent_lab.routing.rules import Rule, match_out_of_scope_rule

__all__ = [
    "IntentResult",
    "classify_with_model",
    "make_guard",
    "Rule",
    "match_out_of_scope_rule",
]
