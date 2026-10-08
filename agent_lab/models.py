"""会话领域的基础数据结构。

Message 是 SessionStore 读写的基本单位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_lab.llm.types import Usage


def utc_now() -> str:
    """返回当前 UTC 时间（ISO 8601 字符串）。"""
    return datetime.now(timezone.utc).isoformat()


# 消息状态：ok = 正常；failed = 这一轮调用没成功，用户输入留下了但没得到回复。
# 组 prompt 时只带 ok 的消息，failed 的留给重试和事后归因。
STATUS_OK = "ok"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class Message:
    """一条会话消息。"""

    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str = field(default_factory=utc_now)
    status: str = STATUS_OK


@dataclass(frozen=True)
class UsageRecord:
    """一次模型调用的用量记录。

    落库而不是只放内存：本项目的前提就是"进程重启后状态还在"，
    统计如果只活在内存里，重启后成本就查不出来了。
    """

    session_id: str
    model: str
    usage: Usage
    created_at: str = field(default_factory=utc_now)