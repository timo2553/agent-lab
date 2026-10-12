"""prompt 注册表。

三个约束，都是为了"提示词改错能在启动期发现，而不是在线上跑偏"：

- 模板放文件里、代码里只按 `(名字, 版本)` 取，改提示词不用改代码；
- 变量集合在**加载期**用正则解析出来，不在渲染时才猜；
- 渲染时多传、少传变量都直接报错——`str.format` 少传会抛 KeyError、
  多传会静默忽略，后者最危险：你以为传进去了，其实模板里根本没这个占位符。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

# 只认 {标识符} 形式的占位符
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptError(RuntimeError):
    """模板缺失、命名不合规或变量对不上。属于配置错误，启动期就该暴露。"""


@dataclass(frozen=True)
class PromptTemplate:
    """一个已加载的模板。`variables` 是从文本里解析出来的占位符集合。"""

    name: str
    version: str
    text: str
    variables: frozenset[str]

    def render(self, **values: str) -> str:
        missing = self.variables - values.keys()
        if missing:
            raise PromptError(
                f"模板 {self.name}.{self.version} 缺少变量：{sorted(missing)}"
            )
        extra = values.keys() - self.variables
        if extra:
            raise PromptError(
                f"模板 {self.name}.{self.version} 用不到这些变量：{sorted(extra)}"
                "（占位符拼错时最常见的表现）"
            )
        return self.text.format(**values)


class PromptRegistry:
    """启动时把 templates/ 下的模板全部加载进来，之后只读。"""

    def __init__(self, templates_dir: Path | str = TEMPLATES_DIR) -> None:
        self._dir = Path(templates_dir)
        self._templates: dict[tuple[str, str], PromptTemplate] = {}
        self._load()

    def _load(self) -> None:
        if not self._dir.is_dir():
            raise PromptError(f"模板目录不存在：{self._dir}")
        for path in sorted(self._dir.glob("*.txt")):
            stem = path.stem  # 例：customer_service_system.v1
            if "." not in stem:
                raise PromptError(f"模板文件名必须是 <名字>.<版本>.txt：{path.name}")
            name, version = stem.rsplit(".", 1)
            key = (name, version)
            if key in self._templates:
                raise PromptError(f"模板重名：{path.name}")
            text = path.read_text(encoding="utf-8")
            self._templates[key] = PromptTemplate(
                name=name,
                version=version,
                text=text,
                variables=frozenset(_PLACEHOLDER.findall(text)),
            )

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """取模板。version 省略时取该名字下版本号最大的一个。"""
        if version is not None:
            template = self._templates.get((name, version))
            if template is None:
                raise PromptError(f"没有这个模板：{name}.{version}")
            return template
        matched = [t for (n, _), t in self._templates.items() if n == name]
        if not matched:
            raise PromptError(f"没有这个模板：{name}")
        return max(matched, key=lambda t: t.version)

    def render(self, name: str, *, version: str | None = None, **values: str) -> str:
        return self.get(name, version).render(**values)

    @property
    def loaded(self) -> list[tuple[str, str]]:
        """已加载的 (名字, 版本)，按名字排序——方便启动期打日志或做自检。"""
        return sorted(self._templates)