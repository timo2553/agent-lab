"""RAG 接线的目标行为：低分兜底不调主模型、命中注入上下文并回填引用。

验收标准（M3）：
- 低分时主模型 call_count == 0，直接回兜底话术；
- 命中时回答携带的引用，能在索引里对上真实片段。
用 FakeLLMClient，既不打网络，结果也可复现。
"""

from __future__ import annotations

from agent_lab.chat.service import ChatService
from agent_lab.llm.fake import FakeLLMClient
from agent_lab.models import STATUS_FAILED
from agent_lab.prompts import PromptRegistry
from agent_lab.rag.chunker import split_document
from agent_lab.rag.corpus import Document
from agent_lab.rag.index import VectorIndex
from agent_lab.rag.retriever import Retriever
from agent_lab.session_store import SessionStore

SHOP = "示例旗舰店"

DOCS = [
    Document(
        doc_id="shipping",
        title="发货与物流",
        text="## 发货时间\n现货商品付款后 24 小时内发货",
    ),
    Document(
        doc_id="returns",
        title="退换货政策",
        text="## 七天无理由退货\n签收后 7 天内可以申请无理由退货",
    ),
]


def make_retriever(min_score: float = 0.15) -> Retriever:
    chunks = []
    for document in DOCS:
        chunks.extend(split_document(document, max_chars=200, overlap=20))
    return Retriever(VectorIndex(chunks), min_score=min_score)


def build(tmp_path, retriever, *script: str | Exception):
    store = SessionStore(tmp_path / "sessions.db")
    client = FakeLLMClient(*script)
    service = ChatService(
        store,
        client,
        registry=PromptRegistry(),
        shop_name=SHOP,
        retriever=retriever,
    )
    return store, client, service


def test_low_score_skips_the_model(tmp_path):
    store, client, service = build(tmp_path, make_retriever())

    reply = service.ask("s1", "量子纠缠黑洞")

    assert client.calls == []                 # 验收：低分兜底时主模型一次都不调
    assert reply.rag_fallback is True
    assert reply.failed is True
    assert reply.sources == ()
    assert SHOP in reply.text                 # 用的是注册表里的固定话术
    assert [(m.content, m.status) for m in store.load_history("s1")] == [
        ("量子纠缠黑洞", STATUS_FAILED),
        (reply.text, STATUS_FAILED),
    ]


def test_hit_injects_context_and_backfills_citations(tmp_path):
    retriever = make_retriever()
    store, client, service = build(tmp_path, retriever, "现货商品付款后 24 小时内发货。")

    reply = service.ask("s1", "多久发货")

    assert len(client.calls) == 1
    assert reply.rag_fallback is False
    assert reply.sources                       # 回填了引用

    # 注入 prompt 的参考资料里，每条引用都能对上索引里的真实片段
    injected = "\n".join(message.content for message in client.calls[0])
    by_id = {chunk.chunk_id: chunk.text for chunk in retriever.index.chunks}
    for citation in reply.sources:
        assert citation.chunk_id in by_id
        assert citation.text == by_id[citation.chunk_id]
        assert citation.text in injected
    assert "【1】" in injected
    assert reply.text == "现货商品付款后 24 小时内发货。"


def test_without_retriever_behaves_like_before(tmp_path):
    """不注入检索器就是 M2 的行为：不检索、直接调模型。"""
    store = SessionStore(tmp_path / "sessions.db")
    client = FakeLLMClient("普通回复")
    service = ChatService(store, client, shop_name=SHOP)

    reply = service.ask("s1", "随便问点什么")

    assert len(client.calls) == 1
    assert reply.sources == ()
    assert reply.rag_fallback is False