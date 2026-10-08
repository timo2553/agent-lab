"""LLM 调用的输入输出数据结构。

这里只放数据，不放传输逻辑（那是 client.py 的事）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Usage:
    """一次调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    """一次调用的结果。

    raw 保留原始返回体：出错排查、以后要做 trace（课程第 37 课）都要靠它。
    """

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)