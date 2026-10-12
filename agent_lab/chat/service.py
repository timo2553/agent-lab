"""最小对话闭环：把会话存储（M0）与模型客户端（M1）接起来。

一轮 ask() 的顺序是固定的：查预算 -> 越界判定 -> 读历史 -> 组 prompt
-> 落库用户输入 -> 调模型 -> 落库回复 -> 记用量。

三个刻意的选择：
- 用户输入**先落库再调用**：调用失败时输入不会丢，只是标成 failed。
- 系统提示词**不落库**：它属于配置，不属于用户对话历史，每次请求时注入。
- 越界判定放在**主模型之前**：判越界就直接回绝，主模型一次都不调。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_lab.cost import BudgetExceededError, summarize
from agent_lab.llm.client import LLMClient
from agent_lab.llm.errors import LLMError
from agent_lab.llm.types import Usage
from agent_lab.models import STATUS_FAILED, STATUS_OK, Message
from agent_lab.prompts import PromptRegistry
from agent_lab.routing import IntentResult
from agent_lab.session_store import SessionStore

# 一进一出的两个可注入点：
# - IntentGuard 判这一轮是否在业务范围内；不注入就退化成"全部放行"（M1 行为）。
# - out_of_scope_text 是越界时的固定话术；不注入就用兜底文案。
IntentGuard = Callable[[str], IntentResult]

# 角色与职责边界（课程第 03 课）：先把"能说什么、不能说什么"钉在系统提示词里。
# 代码层的硬约束（工具白名单、高风险动作审批）在后面几个里程碑补。
DEFAULT_SYSTEM_PROMPT = (
    "你是 agent-lab 的电商客服助手。"
    "只依据对话里已有的信息回答订单、商品与售后问题；"
    "不知道就直说不确定，不要编造订单状态、价格，也不要承诺退款到账时间。"
)


@dataclass(frozen=True)
class Reply:
    """一轮对话的结果。"""

    session_id: str
    text: str
    usage: Usage
    user_message: Message
    assistant_message: Message | None = None
    failed: bool = False
    out_of_scope: bool = False
    intent: str | None = None


class ChatService:
    """最小可用的对话服务：一次问答 + 持久化 + 用量统计。"""

    def __init__(
        self,
        store: SessionStore,
        client: LLMClient,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_limit: int = 20,
        budget_tokens: int | None = None,
        registry: PromptRegistry | None = None,
        guard: IntentGuard | None = None,
        out_of_scope_text: str | Callable[[str], str] | None = None,
        shop_name: str = "本店",
    ) -> None:
        self.store = store
        self.client = client
        self.system_prompt = system_prompt
        self.history_limit = history_limit
        self.budget_tokens = budget_tokens
        self.registry = registry or PromptRegistry()
        self.guard = guard
        self.out_of_scope_text = out_of_scope_text
        self.shop_name = shop_name

    def ask(self, session_id: str, text: str) -> Reply:
        """问一轮，返回这轮的回复与用量；调用失败会抛 LLMError 并留下失败记录。"""
        self.__check_budget(session_id)

        # 越界判定发生在任何模型调用之前：判越界就直接回绝，主模型一次都不调。
        if self.guard is not None:
            result = self.guard(text)
            if not result.in_scope:
                return self.__out_of_scope(session_id, text, result)

        history = self.store.load_history(session_id, limit=self.history_limit, status=STATUS_OK)
        user_message = self.store.append_message(session_id, "user", text)
        prompt = [self.__system_message(session_id), *history, user_message]
        try:
            response = self.client.complete(prompt)
        except LLMError:
            # 失败也要留下用户说过什么，否则用户重发之前的内容就彻底丢了
            self.store.mark_message_status(user_message.message_id, STATUS_FAILED)
            raise

        assistant_message = self.store.append_message(session_id, "assistant", response.text)
        self.store.record_usage(session_id, response.model, response.usage)
        return Reply(
            session_id=session_id,
            text=response.text,
            usage=response.usage,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def session_usage(self, session_id: str) -> tuple[Usage, float | None]:
        """该会话的累计用量与估算费用（从库里读，重启后依然准）。"""
        return summarize(self.store.load_usage(session_id))

    def __system_message(self, session_id: str) -> Message:
        # 借 Message 承载系统提示词，保持 LLMClient 的入参只有一种类型
        return Message(
            message_id=f"system:{session_id}",
            session_id=session_id,
            role="system",
            content=self.system_prompt,
        )

    def __out_of_scope(self, session_id: str, text: str, result: IntentResult) -> Reply:
        """越界：主模型一次不调，回固定话术，两边都落库并标 failed。"""
        user_message = self.store.append_message(session_id, "user", text)
        self.store.mark_message_status(user_message.message_id, STATUS_FAILED)
        reply_text = self.__resolve_out_of_scope_text()
        assistant = self.store.append_message(session_id, "assistant", reply_text)
        self.store.mark_message_status(assistant.message_id, STATUS_FAILED)
        return Reply(
            session_id=session_id,
            text=reply_text,
            usage=Usage(),
            user_message=user_message,
            assistant_message=assistant,
            failed=True,
            out_of_scope=True,
            intent=result.intent,
        )

    def __resolve_out_of_scope_text(self) -> str:
        if isinstance(self.out_of_scope_text, str):
            return self.out_of_scope_text
        if callable(self.out_of_scope_text):
            return self.out_of_scope_text(self.shop_name)
        return self.registry.render("out_of_scope_reply", shop_name=self.shop_name)

    def __check_budget(self, session_id: str) -> None:
        if self.budget_tokens is None:
            return
        used, _ = self.session_usage(session_id)
        if used.total_tokens >= self.budget_tokens:
            raise BudgetExceededError(
                f"会话 {session_id} 累计已用 {used.total_tokens} tokens，"
                f"达到预算上限 {self.budget_tokens}，本轮不再调用模型。"
            )