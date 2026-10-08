"""token 计量与成本估算。"""

from __future__ import annotations

import pytest

from agent_lab.cost import estimate_cost, format_usage, summarize
from agent_lab.llm.types import Usage
from agent_lab.models import UsageRecord
from agent_lab.session_store import SessionStore


def test_estimate_cost_uses_price_table():
    # deepseek-chat 示例价：输入 2 元/百万，输出 8 元/百万
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert estimate_cost("deepseek-chat", usage) == pytest.approx(10.0)


def test_unknown_model_returns_none():
    assert estimate_cost("some-unknown-model", Usage(1, 1)) is None
    assert "费用未知" in format_usage(Usage(1, 1), None)


def test_summarize_adds_up_and_refuses_to_guess():
    records = [
        UsageRecord(session_id="s1", model="deepseek-chat", usage=Usage(100, 50)),
        UsageRecord(session_id="s1", model="deepseek-chat", usage=Usage(100, 50)),
    ]

    total, cost = summarize(records)

    assert (total.prompt_tokens, total.completion_tokens) == (200, 100)
    assert cost == pytest.approx(200 / 1_000_000 * 2 + 100 / 1_000_000 * 8)

    # 只要有一条记录的价格未知，整体费用就标为未知
    _, mixed_cost = summarize([*records, UsageRecord("s1", "mystery-model", Usage(1, 1))])
    assert mixed_cost is None


def test_store_records_usage_and_survives_restart(tmp_path):
    db = tmp_path / "sessions.db"
    store = SessionStore(db)
    store.record_usage("s1", "deepseek-chat", Usage(10, 5))
    store.record_usage("s1", "deepseek-chat", Usage(20, 5))
    store.record_usage("s2", "deepseek-chat", Usage(1, 1))

    reopened = SessionStore(db)

    assert [r.usage.total_tokens for r in reopened.load_usage("s1")] == [15, 25]
    session_total, _ = summarize(reopened.load_usage("s1"))
    assert session_total.total_tokens == 40
    everything, _ = summarize(reopened.load_usage())
    assert everything.total_tokens == 42