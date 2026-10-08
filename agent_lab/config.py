"""运行配置：把模型接入参数从环境变量里读出来。

单独一个模块的原因：从 M1 开始，凡是需要"外部服务参数"的地方都从这里取，
避免 API Key 散落在各个调用点（更不能写死进代码或提交进仓库）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# 环境变量优先用本仓前缀，其次兼容厂商自己的变量名
_KEY_ENVS = ("AGENT_LAB_API_KEY", "DEEPSEEK_API_KEY")
_BASE_URL_ENVS = ("AGENT_LAB_BASE_URL", "DEEPSEEK_BASE_URL")
_MODEL_ENVS = ("AGENT_LAB_MODEL", "DEEPSEEK_MODEL")


class ConfigError(RuntimeError):
    """配置缺失或非法。属于启动期错误，要直接告诉用户怎么修。"""


@dataclass(frozen=True)
class LLMConfig:
    """接入一个 OpenAI 兼容服务所需的最小参数集。"""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = 30.0
    max_retries: int = 2

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMConfig:
        source = os.environ if env is None else env
        api_key = _first(source, _KEY_ENVS)
        if not api_key:
            raise ConfigError(
                "没有找到 API Key。请设置环境变量 AGENT_LAB_API_KEY（或 DEEPSEEK_API_KEY）。\n"
                "如果只想本地演示、不接真实模型，加上 --fake 参数即可，不需要 Key。"
            )
        return cls(
            api_key=api_key,
            base_url=_first(source, _BASE_URL_ENVS) or DEFAULT_BASE_URL,
            model=_first(source, _MODEL_ENVS) or DEFAULT_MODEL,
            timeout_s=float(source.get("AGENT_LAB_TIMEOUT_S") or 30.0),
            max_retries=int(source.get("AGENT_LAB_MAX_RETRIES") or 2),
        )


def _first(env: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None