"""会话领域的基础数据结构。

Message 是 SessionStore 读写的基本单位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> str:
    """返回当前 UTC 时间（ISO 8601 字符串）。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Message:
    """一条会话消息。"""

    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str = field(default_factory=utc_now)
    token_count: int | None = None