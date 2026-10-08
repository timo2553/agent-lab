"""对话闭环的目标行为：历史回灌、失败落库、预算熔断。

用 FakeLLMClient，所以这些用例既不打网络，结果也完全可复现。
"""

from __future__ import annotations

import pytest

from agent_lab.chat.service import DEFAULT_SYSTEM_PROMPT, ChatService
from agent_lab.cost import BudgetExceededError
from agent_lab.llm.errors import LLMError
from agent_lab.llm.fake import FakeLLMClient
from agent_lab.models import STATUS_FAILED
from agent_lab.session_store import SessionStore


def build(tmp_path, *script: str | Exception, **kwargs):
    store = SessionStore(tmp_path / "sessions.db")
    client = FakeLLMClient(*script)
    return store, client, ChatService(store, client, **kwargs)


def test_reply_is_persisted_and_history_is_replayed(tmp_path):
    store, client, service = build(tmp_path, "第一轮回复", "第二轮回复")

    service.ask("s1", "第一问")
    service.ask("s1", "第二问")

    # 第二次请求应当带上：系统提示词 + 第一轮问答 + 本轮提问
    assert [m.role for m in client.calls[1]] == ["system", "user", "assistant", "user"]
    assert [m.content for m in client.calls[1]] == [
        DEFAULT_SYSTEM_PROMPT,
        "第一问",
        "第一轮回复",
        "第二问",
    ]
    assert [m.content for m in store.load_history("s1")] == [
        "第一问",
        "第一轮回复",
        "第二问",
        "第二轮回复",
    ]


def test_system_prompt_is_always_first_and_never_persisted(tmp_path):
    store, client, service = build(tmp_path, "回复")

    service.ask("s1", "问题")

    assert client.calls[0][0].role == "system"
    assert all(m.role != "system" for m in store.load_history("s1"))


def test_conversation_continues_after_restart(tmp_path):
    db = tmp_path / "sessions.db"
    ChatService(SessionStore(db), FakeLLMClient("第一次回复")).ask("s1", "问题")

    client = FakeLLMClient("重启后的回复")
    ChatService(SessionStore(db), client).ask("s1", "接着说")

    assert [m.content for m in client.calls[0]] == [
        DEFAULT_SYSTEM_PROMPT,
        "问题",
        "第一次回复",
        "接着说",
    ]


def test_failed_call_keeps_input_but_excludes_it_from_context(tmp_path):
    store, client, service = build(tmp_path, LLMError("boom"), "补答")

    with pytest.raises(LLMError):
        service.ask("s1", "这句会失败")

    # 输入留下来了（标 failed），但不会进上下文
    assert [(m.content, m.status) for m in store.load_history("s1")] == [
        ("这句会失败", STATUS_FAILED)
    ]
    assert store.load_history("s1", status="ok") == []

    service.ask("s1", "换一句")

    assert [m.content for m in client.calls[1]] == [DEFAULT_SYSTEM_PROMPT, "换一句"]


def test_usage_is_recorded_and_added_up(tmp_path):
    store, _, service = build(tmp_path, "回复内容")

    reply = service.ask("s1", "你好")
    usage, cost = service.session_usage("s1")

    assert usage.prompt_tokens == reply.usage.prompt_tokens
    assert usage.completion_tokens == len("回复内容")
    # fake-model 不在价目表里，费用算不出来时会明确说"未知"，而不是给个错数
    assert cost is None


def test_budget_blocks_the_call_before_spending(tmp_path):
    store, client, service = build(tmp_path, "回复", budget_tokens=5)

    service.ask("s1", "好")  # 预算还没用掉，正常放行
    assert len(client.calls) == 1

    with pytest.raises(BudgetExceededError):
        service.ask("s1", "再问一句")

    # 熔断发生在调用之前：没有第二次请求，也没有多落一条消息
    assert len(client.calls) == 1
    assert [m.content for m in store.load_history("s1")] == ["好", "回复"]