"""提示词注册表：模板与代码分离，按 (名字, 版本) 取用。"""

from agent_lab.prompts.registry import (
    PromptError,
    PromptRegistry,
    PromptTemplate,
    TEMPLATES_DIR,
)

__all__ = ["PromptError", "PromptRegistry", "PromptTemplate", "TEMPLATES_DIR"]