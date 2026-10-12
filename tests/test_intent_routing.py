"""越界判定的目标行为：规则前置、模型分类、fail-closed 兜底、失败落库。

验收标准（M2）：越界请求**主模型调用次数为 0**——先被拦下，才谈得上省钱。
用 FakeLLMClient，既不打网络，结果也完全可复现。
"""

from __future__ import annotations

import pytest

from agent_lab.chat.service import ChatService
from agent_lab.llm.errors import LLMAuthError
from agent_lab.llm.fake import FakeLLMClient
from agent_lab.models import STATUS_FAILED
from agent_lab.prompts import PromptRegistry
from agent_lab.routing import IntentResult, classify_with_model, make_guard
from agent_lab.routing.rules import match_out_of_scope_rule
from agent_lab.session_store import SessionStore

SHOP = "示例旗舰店"

# 解析器测试专用的固定兜底（"判越界"）。和 service 里那套是两回事：
# classifier 只负责返回调用方注入的 fallback，自己不决定策略。
FALLBACK = IntentResult(in_scope=False, intent="fallback", reason="解析失败")


def allow(text: str) -> IntentResult:
    return IntentResult(in_scope=True, intent="in_scope", reason="业务问题")


def block(text: str) -> IntentResult:
    return IntentResult(in_scope=False, intent="rule:test", reason="主题无关")


def build(tmp_path, guard, *script: str | Exception, **kwargs):
    """注意参数顺序：guard 在前，剧本（变长参数）在后。"""
    store = SessionStore(tmp_path / "sessions.db")
    client = FakeLLMClient(*script)
    registry = PromptRegistry()
    service = ChatService(
        store, client, guard=guard, registry=registry, shop_name=SHOP, **kwargs
    )
    return store, client, service


# ---------- 规则前置 ----------

def test_rule_matches_obvious_out_of_scope():
    rule = match_out_of_scope_rule("帮我写一段 Python 代码")
    assert rule is not None
    assert rule.name == "coding"


def test_rule_ignores_normal_business_question():
    assert match_out_of_scope_rule("我的订单什么时候发货？") is None


def test_rule_blocks_before_the_model_is_called():
    client = FakeLLMClient("不该被调用")
    result = make_guard(client, shop_name=SHOP)("帮我写代码")

    assert result.in_scope is False
    assert client.calls == []                    # 规则命中，一次模型调用都没有
    assert "coding" in result.intent             # 归因留痕：是哪条规则挡的


# ---------- 模型分类（练习部分）：解析 -> 校验 -> 兜底 ----------

def classify(text: str, *script: str | Exception) -> tuple[IntentResult, FakeLLMClient]:
    client = FakeLLMClient(*script)
    result = classify_with_model(client, [], fallback=FALLBACK)
    return result, client


def test_parses_plain_json():
    result, _ = classify("查订单", '{"intent": "in_scope", "reason": "业务问题"}')
    assert result.in_scope is True
    assert result.reason == "业务问题"


def test_parses_json_wrapped_in_code_fence():
    script = '好的：\n```json\n{"intent": "out_of_scope", "reason": "在写代码"}\n```'
    result, _ = classify("写代码", script)
    assert result.in_scope is False
    assert result.intent == "out_of_scope"


def test_unparseable_text_falls_back():
    result, _ = classify("随便问", "抱歉我不太确定")
    assert result == FALLBACK


def test_unknown_intent_falls_back():
    result, _ = classify("随便问", '{"intent": "maybe"}')
    assert result == FALLBACK


def test_non_object_json_falls_back():
    result, _ = classify("随便问", '["in_scope"]')
    assert result == FALLBACK


def test_client_failure_propagates_instead_of_falling_back():
    """分类模型调用失败是"服务坏了"，不是"返回不可靠"——不许被兜底吞掉。

    兜底只接住解析/校验失败（ValueError）。网络、鉴权这类 LLMError 必须往上传，
    否则用户会看到"超出服务范围"，而真实原因是服务故障，事后也归因不了。
    """
    client = FakeLLMClient(LLMAuthError("鉴权失败"))
    with pytest.raises(LLMAuthError):
        classify_with_model(client, [], fallback=FALLBACK)


# ---------- service 接线：越界短路 + 失败落库 ----------

def test_out_of_scope_reply_skips_the_model(tmp_path):
    store, client, service = build(tmp_path, block)

    reply = service.ask("s1", "帮我写个爬虫")

    assert client.calls == []                    # 验收：主模型一次都没调
    assert reply.failed is True
    assert "示例旗舰店" in reply.text            # 用的是注册表里的固定话术
    assert reply.assistant_message is not None
    # 用户输入和回绝都落库，且都是失败态（不进后续上下文）
    assert [(m.content, m.status) for m in store.load_history("s1")] == [
        ("帮我写个爬虫", STATUS_FAILED),
        (reply.text, STATUS_FAILED),
    ]


def test_in_scope_still_calls_the_model(tmp_path):
    store, client, service = build(tmp_path, allow, "订单已发货")

    reply = service.ask("s1", "我的订单到哪了？")

    assert len(client.calls) == 1
    assert reply.text == "订单已发货"
    assert reply.failed is False


def test_classifier_failure_reaches_the_caller(tmp_path):
    """端到端：分类器挂了，服务抛出原始错误，而不是给一个"越界"的回复。"""
    classifier = FakeLLMClient(LLMAuthError("鉴权失败"))
    store, main, service = build(tmp_path, make_guard(classifier, shop_name=SHOP))

    with pytest.raises(LLMAuthError):
        service.ask("s1", "我的订单到哪了？")

    assert main.calls == []                  # 主模型没被调用
    assert store.load_history("s1") == []    # 也没留下任何"越界回绝"的记录