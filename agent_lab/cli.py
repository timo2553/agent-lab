"""agent-lab 命令行入口。

    python -m agent_lab ask "订单什么时候到" --session s1
    python -m agent_lab chat --session s1
    python -m agent_lab usage --session s1

加 `--fake` 就用本地假模型：不需要 API Key，也能把整条链路演示一遍。
"""

from __future__ import annotations

import argparse
import os
import sys

from agent_lab.chat.service import ChatService
from agent_lab.config import ConfigError, LLMConfig
from agent_lab.cost import BudgetExceededError, format_usage, summarize
from agent_lab.llm.client import LLMClient, OpenAICompatClient
from agent_lab.llm.errors import LLMError
from agent_lab.llm.fake import FakeLLMClient
from agent_lab.session_store import SessionStore

DEFAULT_DB = "data/sessions.db"
EXIT_COMMANDS = {"exit", "quit", ":q"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "usage":
            return _cmd_usage(args)
        service = _build_service(args)
        if args.command == "ask":
            return _cmd_ask(service, args)
        return _cmd_chat(service, args)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    except BudgetExceededError as exc:
        print(f"[预算熔断] {exc}", file=sys.stderr)
        return 3
    except LLMError as exc:
        # 错误分类到这一层才体现出价值：能不能靠重试解决，提示的话术完全不同
        hint = "可以稍后再试。" if exc.retryable else "重试没有意义，请检查配置，或转人工处理。"
        print(f"[模型调用失败] {exc}\n提示：{hint}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-lab",
        description="从零实现的电商客服 Agent（会话落盘 + 模型接入 + 用量观测）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="问一轮并打印回复")
    ask.add_argument("text", help="用户输入")
    ask.add_argument("--session", default="default", help="会话 id")
    _add_dialog_options(ask)

    chat = sub.add_parser("chat", help="进入连续对话（REPL）")
    chat.add_argument("--session", default="default", help="会话 id")
    _add_dialog_options(chat)

    usage = sub.add_parser("usage", help="查看累计 token 与估算费用")
    usage.add_argument("--session", default=None, help="只统计某个会话；不填则统计全部")
    _add_db_option(usage)
    return parser


def _add_db_option(parser: argparse.ArgumentParser) -> None:
    # --db 放在子命令之后，位置符合直觉：ask "..." --session s1 --fake --db xxx
    parser.add_argument(
        "--db",
        default=os.environ.get("AGENT_LAB_DB", DEFAULT_DB),
        help=f"会话库路径（默认 {DEFAULT_DB}，也可用环境变量 AGENT_LAB_DB）",
    )


def _add_dialog_options(parser: argparse.ArgumentParser) -> None:
    _add_db_option(parser)
    parser.add_argument("--fake", action="store_true", help="用本地假模型，不需要 API Key")
    parser.add_argument("--budget", type=int, default=None, help="单会话 token 预算上限")
    parser.add_argument("--limit", type=int, default=20, help="回灌给模型的最大历史条数")


def _open_store(args: argparse.Namespace) -> SessionStore:
    return SessionStore(args.db)


def _build_service(args: argparse.Namespace) -> ChatService:
    client: LLMClient
    if args.fake:
        client = FakeLLMClient("（fake）您好，我这边是演示用的假模型，已经收到您的消息。")
    else:
        client = OpenAICompatClient(LLMConfig.from_env())
    return ChatService(
        _open_store(args),
        client,
        history_limit=args.limit,
        budget_tokens=args.budget,
    )


def _cmd_ask(service: ChatService, args: argparse.Namespace) -> int:
    reply = service.ask(args.session, args.text)
    print(reply.text)
    usage, cost = service.session_usage(args.session)
    print(f"[本次] total={reply.usage.total_tokens} tokens")
    print(f"[会话累计] {format_usage(usage, cost)}")
    return 0


def _cmd_chat(service: ChatService, args: argparse.Namespace) -> int:
    print(f"会话 {args.session} 已开始。输入 exit 退出，Ctrl+C 中断。")
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            return 0
        try:
            reply = service.ask(args.session, text)
        except LLMError as exc:
            # 单轮失败不终止会话：输入已落库标为 failed，用户可以重发
            print(f"[本轮失败] {exc}（已记录，可重试）", file=sys.stderr)
            continue
        except BudgetExceededError as exc:
            print(f"[预算熔断] {exc}", file=sys.stderr)
            return 3
        print(f"客服> {reply.text}")
        usage, cost = service.session_usage(args.session)
        print(f"       [会话累计 {format_usage(usage, cost)}]")


def _cmd_usage(args: argparse.Namespace) -> int:
    records = _open_store(args).load_usage(args.session)
    usage, cost = summarize(records)
    scope = args.session or "全部会话"
    print(f"[{scope}] 调用 {len(records)} 次 | {format_usage(usage, cost)}")
    return 0