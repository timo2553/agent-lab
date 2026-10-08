"""LLM 客户端的契约与错误分类。

全部离线：传输函数是注入的假 opener，一个真实请求都不会发出去，
所以这套用例在 CI 里永远能跑，也不花一分钱。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from agent_lab.config import LLMConfig
from agent_lab.llm.client import OpenAICompatClient
from agent_lab.llm.errors import (
    LLMAuthError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
)
from agent_lab.models import Message

OK_BODY = json.dumps(
    {
        "model": "deepseek-chat",
        "choices": [
            {"message": {"role": "assistant", "content": "您好"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
).encode("utf-8")


class FakeResponse:
    """够用的 urlopen 返回值：支持 read() 和 with。"""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def make_opener(*results: bytes | Exception):
    """按顺序返回结果，用完后重复最后一项；同时记录每次请求供断言。"""
    seen: list[tuple[urllib.request.Request, float | None]] = []

    def opener(request: urllib.request.Request, timeout: float | None = None):
        seen.append((request, timeout))
        item = results[min(len(seen) - 1, len(results) - 1)]
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    return opener, seen


def http_error(code: int, payload: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.com", code, "err", {}, io.BytesIO(payload.encode("utf-8"))
    )


def make_client(opener, *, max_retries: int = 2, sleep=None) -> OpenAICompatClient:
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="deepseek-chat",
        timeout_s=5.0,
        max_retries=max_retries,
    )
    return OpenAICompatClient(config, opener=opener, sleep=sleep or (lambda _s: None))


def user_message(text: str = "你好") -> Message:
    return Message(message_id="m1", session_id="s1", role="user", content=text)


def test_request_payload_and_auth_header():
    opener, seen = make_opener(OK_BODY)
    client = make_client(opener)

    client.complete([user_message()])

    request, timeout = seen[0]
    assert request.full_url == "https://example.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert timeout == 5.0
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
    }


def test_parses_text_usage_and_model():
    opener, _ = make_opener(OK_BODY)

    response = make_client(opener).complete([user_message()])

    assert response.text == "您好"
    assert response.model == "deepseek-chat"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 18
    assert response.finish_reason == "stop"


def test_auth_error_is_not_retried():
    opener, seen = make_opener(http_error(401, '{"error":"invalid key"}'))

    with pytest.raises(LLMAuthError) as excinfo:
        make_client(opener).complete([user_message()])

    assert excinfo.value.status_code == 401
    assert excinfo.value.retryable is False
    assert len(seen) == 1  # 重试没有意义，只应该调用一次


def test_rate_limit_is_retried_with_backoff():
    sleeps: list[float] = []
    opener, seen = make_opener(http_error(429, "slow down"), OK_BODY)

    response = make_client(opener, sleep=sleeps.append).complete([user_message()])

    assert response.text == "您好"
    assert len(seen) == 2
    assert sleeps == [0.5]  # 第一次重试退避 0.5s


def test_server_error_retries_then_raises():
    opener, seen = make_opener(http_error(503, "unavailable"))

    with pytest.raises(LLMServerError):
        make_client(opener, max_retries=2).complete([user_message()])

    assert len(seen) == 3  # 首次 + 2 次重试


def test_timeout_retries_then_raises_timeout_error():
    opener, seen = make_opener(TimeoutError("timed out"))

    with pytest.raises(LLMTimeoutError):
        make_client(opener, max_retries=1).complete([user_message()])

    assert len(seen) == 2


def test_url_error_with_timeout_reason_is_timeout():
    opener, _ = make_opener(urllib.error.URLError(TimeoutError("timed out")))

    with pytest.raises(LLMTimeoutError):
        make_client(opener, max_retries=0).complete([user_message()])


def test_network_error_is_classified_and_retried():
    opener, seen = make_opener(urllib.error.URLError("dns failure"))

    with pytest.raises(LLMNetworkError):
        make_client(opener, max_retries=1).complete([user_message()])

    assert len(seen) == 2


def test_non_json_body_raises_format_error():
    opener, seen = make_opener(b"<html>502 Bad Gateway</html>")

    with pytest.raises(LLMResponseFormatError):
        make_client(opener).complete([user_message()])

    assert len(seen) == 1  # 返回体格式不对，重试也白搭


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"choices": []}).encode("utf-8"),
        json.dumps({"choices": [{"message": {}}]}).encode("utf-8"),
        json.dumps(["not", "an", "object"]).encode("utf-8"),
    ],
)
def test_malformed_body_raises_format_error(body):
    opener, _ = make_opener(body)

    with pytest.raises(LLMResponseFormatError):
        make_client(opener).complete([user_message()])