"""老库迁移：M0 时期建的库没有 status 列，新代码必须能直接读。

CREATE TABLE IF NOT EXISTS 不会改动已存在的表，所以加字段一定要显式迁移。
"""

from __future__ import annotations

import sqlite3

from agent_lab.session_store import SessionStore

LEGACY_SCHEMA = """
CREATE TABLE messages(
seq             INTEGER PRIMARY KEY AUTOINCREMENT,
message_id      TEXT NOT NULL,
session_id      TEXT NOT NULL,
role            TEXT NOT NULL,
content         TEXT NOT NULL,
created_at      TEXT NOT NULL
)
"""


def make_legacy_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO messages(message_id, session_id, role, content, created_at)"
        "VALUES (?, ?, ?, ?, ?)",
        ("m1", "s1", "user", "老数据", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()


def test_legacy_database_is_migrated_on_open(tmp_path):
    db = tmp_path / "legacy.db"
    make_legacy_db(db)

    store = SessionStore(db)

    assert [(m.content, m.status) for m in store.load_history("s1")] == [("老数据", "ok")]


def test_migrated_database_accepts_new_writes_with_status(tmp_path):
    db = tmp_path / "legacy.db"
    make_legacy_db(db)
    store = SessionStore(db)

    store.append_message("s1", "assistant", "新回复")
    store.mark_message_status("m1", "failed")

    assert [(m.content, m.status) for m in store.load_history("s1")] == [
        ("老数据", "failed"),
        ("新回复", "ok"),
    ]