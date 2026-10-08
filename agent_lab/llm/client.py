"""OpenAI 兼容协议的 LLM 客户端。

【这一步由你自己实现】目标：让 tests/test_llm_client.py 全绿。

- 参考答案在 ../reference/llm_client_reference.py（已进 .gitignore，不会推到 GitHub）。
  建议先自己写，卡住了再去看对应那一个方法的实现。
- 不要改测试：测试就是这一步的需求说明。
- 建议实现顺序：_build_request -> _parse -> complete/_send（重试与错误分类在最后）。
- 每写完一个方法就跑一次对应用例，比一口气写完再调试快得多：

      python -m pytest tests/test_llm_client.py -x -q
      （Windows 下用 .venv 里的解释器：.venv/Scripts/python -m pytest ...）
"""

from __future__ import annotations

import time
import json
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from agent_lab.config import LLMConfig
from agent_lab.llm.errors import (
    LLMError,
    LLMNetworkError,
    LLMResponseFormatError,
    LLMTimeoutError,
    classify_status,
)
from agent_lab.llm.types import LLMResponse, Usage
from agent_lab.models import Message

# 传输函数可注入：测试里塞一个假的进去，就不会打真实网络（这部分已给定）
Opener = Callable[..., Any]


class LLMClient(Protocol):
    """所有模型接入都必须满足的契约。已给定，不用改。"""

    def complete(self, messages: list[Message]) -> LLMResponse: ...


class OpenAICompatClient:
    """POST {base_url}/chat/completions，一次性拿到完整回复（不做流式）。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        opener: Opener | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep

    def complete(self, messages: list[Message]) -> LLMResponse:
        """拿到一次回复；可重试的错误要重试，其余直接抛。

        验收：test_rate_limit_is_retried_with_backoff、test_server_error_retries_then_raises、
              test_timeout_retries_then_raises_timeout_error、test_auth_error_is_not_retried

        提示：
        - 请求对象构造**一次**，重试时复用（构造放在循环外）
        - 只重试 exc.retryable 为真、且还有重试次数的错误；其余立刻抛出去
        - 退避用 self._sleep(0.5 * 2**attempt)：第一次重试等 0.5s，第二次 1s
        - 不要吞掉最后一次失败——它必须抛到调用方
        """
        request = self._build_request(messages)

        last_error: LLMError | None = None
        for attempt in range(self.config.max_retries+1):
            try:
                return self._send(request)
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                if attempt < self.config.max_retries:
                    self._sleep(0.5 * 2**attempt)
        assert last_error is not None
        raise last_error

    def _build_request(self, messages: list[Message]) -> urllib.request.Request:
        """构造请求：URL + JSON body + 两个 header。
        """
        payload = {
            "model": self.config.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            self.config.endpoint,
            data=data, 
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.config.api_key}"
                    },
                    method="POST",
                    )

    def _send(self, request: urllib.request.Request) -> LLMResponse:
        """发出请求并解析；把 urllib 的异常翻译成我们自己的错误类型。
        """
        try:
            with self._opener(request, timeout=self.config.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            text = raw.decode("utf-8",errors="replace") if isinstance(raw,bytes) else str(raw)
            raise classify_status(exc.code, text) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LLMTimeoutError(f"读取超时：{exc.reason}")  from exc
            raise LLMNetworkError(f"网络错误：{exc.reason}")  from exc
        except TimeoutError as exc:
            raise  LLMTimeoutError(f"请求超时：{exc}")  from exc
        return self._parse(body)

    def _parse(self, body: bytes) -> LLMResponse:
        """把返回体变成 LLMResponse。
        """
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMResponseFormatError(f"返回不是合法json:{body[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMResponseFormatError("返回不是json对象")
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise LLMResponseFormatError("返回体缺少choices字段")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseFormatError("choices[0]不是对象")
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseFormatError("choices[0].message.content不是字符串")
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        return LLMResponse(
            text=content,
            model=data.get("model") or self.config.model,   # 缺失 / 空串都退回配置
            usage=Usage(
                prompt_tokens=_int_or_zero(usage.get("prompt_tokens")),
                completion_tokens=_int_or_zero(usage.get("completion_tokens")),
            ),
            finish_reason=first.get("finish_reason"),
            raw=data,                                       # 整个返回体留给 trace
        )
def _int_or_zero(value: object) -> int:
    """usage 字段缺失 / null / 类型不对时一律按 0 处理（bool 是 int 子类，要排除）。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
