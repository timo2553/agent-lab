"""集成测试：client 走真实 socket 的完整链路。

和 tests/test_llm_client.py 的分工见 tests/http_stub.py 顶部。
本文件里 socket 是真的，所以能验到"超时是不是真的生效""重试是不是真的发了第二次请求"。

===== 练习 =====
把下面 5 个用例的 NotImplementedError 换成真实断言。规则：
- 不要改 tests/http_stub.py，它是工具
- 不要改 agent_lab/ 下的实现，用例就是需求
- 每个用例的 docstring 写清了目标、断言和可用的工具函数，照着做就行
"""

from __future__ import annotations

import pytest

from agent_lab.config import LLMConfig
from agent_lab.llm.client import OpenAICompatClient
from agent_lab.llm.errors import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTimeoutError,
)
from agent_lab.models import Message
from http_stub import StubServer, http_error, ok, raw

PAYLOAD = {
    "model": "stub-model",
    "choices": [
        {
            "message": {"role": "assistant", "content": "已发货，明天到"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 21, "completion_tokens": 13},
}


def make_client(stub: StubServer, *, max_retries: int = 0, timeout_s: float = 5.0) -> OpenAICompatClient:
    """接真实 socket 的 client。

    只把退避换成了空函数——否则重试用例会真的睡 0.5s；
    opener 保持默认（真的 urllib.request.urlopen）。
    """
    config = LLMConfig(
        api_key="stub-key",
        base_url=stub.base_url,
        model="stub-model",
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    return OpenAICompatClient(config, sleep=lambda _seconds: None)


def user_message(text: str = "订单到哪了？") -> Message:
    return Message(message_id="m1", session_id="s1", role="user", content=text)


def test_roundtrip_sends_expected_request_and_parses_reply():
    """一次成功往返：请求发成什么样、返回怎么解析。

    提示：
    - 用 `with StubServer(lambda record: ok(PAYLOAD)) as stub:` 起桩服务
    - 请求：`record = stub.requests[0]`，断言
        record.path == "/v1/chat/completions"
        record.headers.get("Authorization") == "Bearer stub-key"
    - 请求体：`payload = record.json_body`，断言 model / messages / stream
      （注意 messages 是 OpenAI 格式：[{"role": ..., "content": ...}]）
    - 返回：断言 response.text 和 response.usage.total_tokens
    - header 的键大小写按客户端实际发来的原样保存，取值用 .get("Authorization")
    """
    with StubServer( lambda record: ok(PAYLOAD)) as stub: 
        client = make_client(stub)
        response = client.complete([user_message()])
        assert response.text == "已发货，明天到"
        assert response.usage.total_tokens == 34
        record = stub.requests[0]
        assert record.path == "/v1/chat/completions"
        assert record.headers.get("Authorization") == "Bearer stub-key"
        payload = record.json_body
        assert payload["model"] == "stub-model"
        assert payload["messages"] == [{"role": "user", "content": "订单到哪了？"}]
        assert payload["stream"] is False


def test_retryable_status_is_retried_once():
    """429 属于可重试：max_retries=1 时应该发 2 次请求，最后仍抛 LLMRateLimitError。

    提示：
    - responder 一直回 `http_error(429, "rate limited")`
    - `with pytest.raises(LLMRateLimitError) as excinfo:` 接住异常
    - 断言 `stub.call_count == 2`（首次 + 1 次重试）和 `excinfo.value.retryable is True`
    - 注意 max_retries=0 是默认值，这个用例要显式传 1
    """

    with StubServer(lambda record: http_error(429, "rate limited")) as stub:
        client = make_client(stub, max_retries=1)
        with pytest.raises(LLMRateLimitError) as excinfo:
            client.complete([user_message()])
        assert stub.call_count == 2
        assert excinfo.value.retryable is True  


def test_auth_error_is_not_retried():
    """401 不可重试：只应该发 1 次请求。

    提示：`http_error(401, "invalid key")`；断言抛 LLMAuthError 且 `stub.call_count == 1`
    """
    with StubServer(lambda record: http_error(401, "invalid key")) as stub:
        client = make_client(stub)
        with pytest.raises(LLMAuthError) as excinfo:
            client.complete([user_message()])
        assert stub.call_count == 1
        assert excinfo.value.retryable is False


def test_non_json_reply_raises_format_error():
    """200 但返回体不是 JSON（网关吐了 HTML）：格式错误，且不该重试。

    提示：用 `raw("<html>502 Bad Gateway</html>")`；断言抛 LLMResponseFormatError 且只请求了 1 次
    """
    with StubServer(lambda record: raw("<html>502 Bad Gateway</html>")) as stub:
        client = make_client(stub)
        with pytest.raises(LLMResponseFormatError) as excinfo:
            client.complete([user_message()])
        assert stub.call_count == 1
        assert excinfo.value.retryable is False


def test_slow_reply_raises_timeout():
    """服务端比客户端超时还慢：应该抛 LLMTimeoutError。

    提示：
    - `ok(PAYLOAD, delay_s=1.5)` 配 `timeout_s=0.3`，让用例能在 1 秒内跑完
    - 断言抛 LLMTimeoutError
    - 桩服务在客户端走掉之后写 socket 会失败，http_stub 已经吞掉了这类 OSError，
      所以你不需要为它做任何事
    """
    with StubServer(lambda record: ok(PAYLOAD, delay_s=1.5)) as stub:
        client = make_client(stub, timeout_s=0.3)
        with pytest.raises(LLMTimeoutError) as excinfo:
            client.complete([user_message()])
        assert stub.call_count == 1
        assert excinfo.value.retryable is True