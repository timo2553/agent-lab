"""SessionStore 的目标行为。

实现之前这些测试全部失败，这是故意的（红 -> 绿）。
不要为了让它变绿而改测试——改 agent_lab/session_store.py。

tmp_path 是 pytest 提供的临时目录 fixture，每个测试一个全新的空目录，
所以测试之间不会互相污染，也不会在你桌面上留下垃圾文件。
"""

from __future__ import annotations

from agent_lab.session_store import SessionStore


def test_append_returns_the_stored_message(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")

    stored = store.append_message("s1", "user", "你好")

    assert stored.role == "user"
    assert stored.content == "你好"


def test_append_then_load_returns_messages_in_order(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.append_message("s1", "user", "你们家耳机怎么样？")
    store.append_message("s1", "assistant", "您好，请问您关注哪方面？")

    history = store.load_history("s1")

    assert [m.role for m in history] == ["user", "assistant"]
    assert [m.content for m in history] == ["你们家耳机怎么样？", "您好，请问您关注哪方面？"]


def test_history_survives_restart(tmp_path):
    """核心用例：换一个全新实例指向同一路径，相当于进程重启。"""
    db = tmp_path / "sessions.db"
    first = SessionStore(db)
    first.append_message("s1", "user", "订单什么时候到？")

    second = SessionStore(db)
    history = second.load_history("s1")

    assert len(history) == 1
    assert history[0].content == "订单什么时候到？"


def test_sessions_are_isolated(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.append_message("s1", "user", "问题一")
    store.append_message("s2", "user", "问题二")

    assert [m.content for m in store.load_history("s1")] == ["问题一"]
    assert [m.content for m in store.load_history("s2")] == ["问题二"]


def test_limit_keeps_the_latest_messages_in_chronological_order(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    for i in range(5):
        store.append_message("s1", "user", f"第 {i} 条")

    history = store.load_history("s1", limit=2)

    assert [m.content for m in history] == ["第 3 条", "第 4 条"]


def test_load_unknown_session_returns_empty_list(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")

    assert store.load_history("never-existed") == []


def test_delete_session_removes_only_that_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store.append_message("s1", "user", "问题一")
    store.append_message("s2", "user", "问题二")

    store.delete_session("s1")

    assert store.load_history("s1") == []
    assert len(store.load_history("s2")) == 1