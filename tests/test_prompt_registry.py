"""prompt 注册表：加载、取版本、变量校验。

这一层的价值在"改错能早发现"：模板名写错、占位符拼错、变量少传，
都必须在启动期或调用点立刻报错，而不是把一段没替换的文本发给用户。
"""

from __future__ import annotations

import pytest

from agent_lab.prompts import PromptError, PromptRegistry


@pytest.fixture
def registry() -> PromptRegistry:
    """用仓库里真实的模板文件。"""
    return PromptRegistry()


def test_loads_repo_templates(registry: PromptRegistry) -> None:
    assert ("customer_service_system", "v1") in registry.loaded
    assert ("out_of_scope_reply", "v1") in registry.loaded


def test_render_replaces_variables(registry: PromptRegistry) -> None:
    text = registry.render("out_of_scope_reply", shop_name="示例旗舰店")
    assert "示例旗舰店" in text
    assert "{shop_name}" not in text


def test_system_prompt_declares_scope(registry: PromptRegistry) -> None:
    """系统提示词必须显式声明范围——越界回绝的第一道（软）约束。"""
    text = registry.render("customer_service_system", shop_name="示例旗舰店")
    assert "超出" in text or "范围" in text


def test_missing_variable_raises(registry: PromptRegistry) -> None:
    with pytest.raises(PromptError) as excinfo:
        registry.render("out_of_scope_reply")
    assert "shop_name" in str(excinfo.value)


def test_extra_variable_raises(registry: PromptRegistry) -> None:
    """多传变量必须报错：多半是占位符拼错了，静默忽略最危险。"""
    with pytest.raises(PromptError):
        registry.render("out_of_scope_reply", shop_name="示例旗舰店", shop_nmae="x")


def test_unknown_name_raises(registry: PromptRegistry) -> None:
    with pytest.raises(PromptError):
        registry.get("not_exist")


def test_unknown_version_raises(registry: PromptRegistry) -> None:
    with pytest.raises(PromptError):
        registry.get("out_of_scope_reply", "v99")


def test_bad_filename_is_rejected(tmp_path) -> None:
    """文件名必须是 <名字>.<版本>.txt，否则加载期直接报错。"""
    (tmp_path / "no_version.txt").write_text("随便写点", encoding="utf-8")
    with pytest.raises(PromptError):
        PromptRegistry(tmp_path)


def test_latest_version_wins(tmp_path) -> None:
    """不指定版本时取最大的版本号。"""
    (tmp_path / "greet.v1.txt").write_text("旧", encoding="utf-8")
    (tmp_path / "greet.v2.txt").write_text("新", encoding="utf-8")
    assert PromptRegistry(tmp_path).render("greet") == "新"