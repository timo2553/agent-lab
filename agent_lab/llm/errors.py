"""LLM 调用的错误分类。

分类只服务于两个判断：这次失败**能不能靠重试解决**、**要不要让用户稍后再试**。
全部当成一个 Exception 一把抓，是做不出能上线的客服的。
"""

from __future__ import annotations


class LLMError(RuntimeError):
    """LLM 调用失败。retryable 表示这次失败能否靠重试解决。"""

    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMAuthError(LLMError):
    """401/403：Key 不对或没权限。重试没有意义，要改配置。"""


class LLMRateLimitError(LLMError):
    """429：被限流。等一会儿再试有意义。"""

    retryable = True


class LLMServerError(LLMError):
    """5xx：服务端抖动。重试有意义。"""

    retryable = True


class LLMTimeoutError(LLMError):
    """读超时。重试有意义。"""

    retryable = True


class LLMNetworkError(LLMError):
    """连不上（DNS / 代理 / 断网）。重试有意义。"""

    retryable = True


class LLMResponseFormatError(LLMError):
    """返回体不是预期结构（HTML 错误页、缺少 choices 等）。重试通常没用。"""


def classify_status(status_code: int, body: str) -> LLMError:
    """按 HTTP 状态码归类，并保留一小段服务端返回内容方便排查。"""
    snippet = body.strip().replace("\n", " ")[:200]
    if status_code in (401, 403):
        return LLMAuthError(f"鉴权失败({status_code})：{snippet}", status_code=status_code)
    if status_code == 429:
        return LLMRateLimitError(f"被限流(429)：{snippet}", status_code=status_code)
    if status_code >= 500:
        return LLMServerError(f"服务端错误({status_code})：{snippet}", status_code=status_code)
    return LLMError(f"请求被拒绝({status_code})：{snippet}", status_code=status_code)