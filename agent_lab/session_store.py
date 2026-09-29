"""会话与消息的持久化存储。

让会话状态在 Agent 进程重启后依然可用，替代课程里进程内字典的方案。

存储形态选了 SQLite（取舍见 README「设计决策 1」）：
- 一个 SessionStore 实例对应一个库文件，按 session_id 隔离
- 开启 WAL + 10s 等锁超时，避免并发写直接抛 database is locked（见「设计决策 3」）
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path

from agent_lab.models import Message

# 建表 + 切 WAL 是"一次性独占"操作：切换 journal_mode 时 SQLite 不走 busy 等待，
# 并发初始化会直接抛 database is locked，所以进程内用一把锁串行化。
_INIT_LOCK = threading.Lock()


class SessionStore:
    """按 session_id 隔离的会话消息存储。

    一个 SessionStore 实例对应一个存储位置。测试会在同一个路径上
    先后创建两个实例来模拟"进程重启"，所以不要把数据只放在内存里。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.__init__db()

    def __connect_db(self) -> sqlite3.Connection:
        """连接数据库，并统一设置等锁超时。

        sqlite3.connect 的 timeout 是"拿不到锁时最多等多久"，不是连接存活时间。
        这里显式设 10s，并同步设置 busy_timeout，让并发写排队而不是直接报错。
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def __init__db(self) -> None:
        """初始化数据库。"""
        with _INIT_LOCK:
            conn = self.__connect_db()
            try:
                # WAL 模式下读不阻塞写、写不阻塞读，配合 busy_timeout 支撑并发读写
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages(
                    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id      TEXT NOT NULL,
                    session_id      TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def append_message(self, session_id: str, role: str, content: str) -> Message:
        msg = Message(
            message_id=uuid.uuid4().hex,
            session_id=session_id,
            role=role,
            content=content,
        )
        conn = self.__connect_db()
        try:
            conn.execute(
                "INSERT INTO messages"
                "(message_id, session_id, role, content, created_at)"
                "VALUES (?, ?, ?, ?, ?)",
                (
                    msg.message_id,
                    msg.session_id,
                    msg.role,
                    msg.content,
                    msg.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return msg

    def load_history(self, session_id: str, limit: int = 20) -> list[Message]:
        """按时间正序返回该会话最近的 limit 条消息。

        语义（测试按这个来，别改）：
        - 只取最近的 limit 条，但返回值仍按时间正序排列（旧 -> 新）
        - 会话不存在、或没有任何消息时，返回空列表，不要抛异常
        """
        if limit <= 0:
            return []
        conn = self.__connect_db()
        try:
            rows = conn.execute(
                "SELECT message_id, role, content, created_at FROM messages "
                "WHERE session_id=? ORDER BY seq DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        finally:
            conn.close()
        rows.reverse()
        return [
            Message(
                message_id=r["message_id"],
                session_id=session_id,
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """删除该会话的全部消息，不影响其他会话。

        语义（测试按这个来，别改）：
        - 删除不存在的会话不是错误，静默返回即可
        """
        conn = self.__connect_db()
        try:
            conn.execute(
                "DELETE FROM messages "
                "WHERE session_id=?",
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()