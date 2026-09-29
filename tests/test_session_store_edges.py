"""自己补的边界用例。

课程给的 7 个用例覆盖了主流程，这里补的是它们没碰到的两处：
并发写同一会话，以及重复删除。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agent_lab.session_store import SessionStore


def test_concurrent_writes_all_land(tmp_path):
    """两个线程同时往同一个 session 写，20 条都不能丢、也不能抛异常。

    这是 README「设计决策 3」的验证用例：靠 WAL + busy_timeout，
    写请求排队而不是直接抛 database is locked。
    """
    db = tmp_path / "sessions.db"

    def write_batch(prefix: str) -> None:
        store = SessionStore(db)  # 每个线程独立连接
        for i in range(10):
            store.append_message("s1", "user", f"{prefix}-{i}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write_batch, ["a", "b"]))

    history = SessionStore(db).load_history("s1", limit=100)

    assert len(history) == 20
    assert {m.content for m in history} == (
        {f"a-{i}" for i in range(10)} | {f"b-{i}" for i in range(10)}
    )


def test_delete_is_idempotent(tmp_path):
    """删两次不报错，且不影响别的会话。"""
    store = SessionStore(tmp_path / "sessions.db")
    store.append_message("s1", "user", "问题一")
    store.append_message("s2", "user", "问题二")

    store.delete_session("s1")
    store.delete_session("s1")

    assert store.load_history("s1") == []
    assert len(store.load_history("s2")) == 1