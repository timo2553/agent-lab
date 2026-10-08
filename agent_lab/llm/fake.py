"""测试与离线演示用的假模型。

它实现 LLMClient 契约，只按剧本返回，不发任何网络请求。
CLI 的 --fake 参数就是靠它——没有 API Key 也能把整条链路演示一遍。
"""

from __future__ import annotations

from agent_lab.llm.types import LLMResponse, Usage
from agent_lab.models import Message

ScriptItem = str | LLMResponse | Exception


class FakeLLMClient:
    """按剧本返回的假客户端。

    剧本里每一项可以是：
    - str：当成回复文本
    - LLMResponse：原样返回
    - Exception：抛出来，用来测错误分支

    剧本用完后重复最后一条，方便 REPL 里连续演示。
    usage 按文本长度估算，保证测试里可复现、也能驱动预算熔断。
    """

    def __init__(self, *script: ScriptItem, model: str = "fake-model") -> None:
        self._script = list(script)
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResponse:
        self.calls.append(list(messages))
        item = self._next_item()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(
            text=item,
            model=self.model,
            usage=Usage(
                prompt_tokens=sum(len(m.content) for m in messages),
                completion_tokens=len(item),
            ),
            finish_reason="stop",
        )

    def _next_item(self) -> ScriptItem:
        if not self._script:
            return "（fake 没有剧本了）"
        if len(self._script) == 1:
            return self._script[0]
        return self._script.pop(0)