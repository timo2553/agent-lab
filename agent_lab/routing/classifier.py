"""两段式越界判定 —— 第二段：让模型做意图分类（练习部分）。

第一段是 rules.py：免费、不延迟的硬规则，命中直接判越界。
只有没命中规则、拿不准的问题，才走到这里花 token 请模型判断。

模型按内部协议返回 JSON，本模块负责把返回文本变成 IntentResult：
**解析 -> 校验 -> 兜底**。解析或校验任何一步失败，都返回调用方注入的
fallback（fail-closed：判定不可靠时宁可挡在外面，也不要放行）。

这个文件是你的练习部分：classify_with_model 留了签名和约束，
实现要点写在 docstring 里。不要改 IntentResult 和 _INTENTS。
"""

from __future__ import annotations
import json
from dataclasses import dataclass

from agent_lab.llm.client import LLMClient
from agent_lab.models import Message

# 模型必须二选一返回的意图。它出现在分类提示词里，也出现在校验集合里。
_INTENTS = ("in_scope", "out_of_scope")


@dataclass(frozen=True)
class IntentResult:
   """一次意图判定的结果。in_scope=False 表示这一轮不走主模型。"""

   in_scope: bool
   intent: str
   reason: str


def classify_with_model(
   client: LLMClient,
   messages: list[Message],
   *,
   fallback: IntentResult,
) -> IntentResult:
   """调模型做意图分类，把模型返回的文本变成 IntentResult。
    """
   try:
      response = client.complete(messages)
      payload = json.loads(_extract_json_object(response.text))
   except ValueError:
      return fallback
   if not isinstance(payload,dict):
      return fallback
   intent = payload.get("intent")
   if intent not in _INTENTS:
      return fallback
   reason = payload.get("reason")
   if not isinstance(reason,str):
      return fallback
   return IntentResult(
      in_scope=(intent == "in_scope"),
      intent=intent,
      reason=reason,
   )
def _extract_json_object(text: str) -> str:
   """从文本中提取 JSON 字符串。"""
   stripped = text.strip()
   if "```" in stripped:
      chunks = stripped.split("```")
      if len(chunks) >= 2:
         inner = chunks[1].strip()
         if inner[:4].lower() == "json":
            inner = inner[4:].strip()
         if inner:
            stripped = inner
   start = stripped.find("{")
   end = stripped.rfind("}")
   if start == -1 or end == -1 or start > end:
      raise ValueError("模型输出没有包含 JSON 对象")
   return stripped[start:end+1]

