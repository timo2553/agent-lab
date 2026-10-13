"""文档切分的目标行为：语义优先 + 长度兜底。

用自己构造的小 Document，块数完全可控；最后再用真实语料跑一遍，确保切出来的
块都在长度预算内、doc_id 也对得上。
"""

from __future__ import annotations

from agent_lab.rag.chunker import split_document
from agent_lab.rag.corpus import Document, load_corpus


def doc(text: str, *, doc_id: str = "d1", title: str = "测试文档") -> Document:
    return Document(doc_id=doc_id, title=title, text=text)


def test_splits_on_second_level_headings():
    d = doc("开头说明\n## 第一节\n内容一\n## 第二节\n内容二")
    chunks = split_document(d, max_chars=100, overlap=10)
    assert [c.text for c in chunks] == [
        "开头说明",
        "## 第一节\n内容一",
        "## 第二节\n内容二",
    ]


def test_chunk_ids_are_sequential_and_carry_metadata():
    d = doc("## 一\n甲\n## 二\n乙", doc_id="shipping", title="发货与物流")
    chunks = split_document(d, max_chars=100, overlap=10)
    assert [c.chunk_id for c in chunks] == ["shipping#0", "shipping#1"]
    assert all(c.doc_id == "shipping" and c.title == "发货与物流" for c in chunks)


def test_short_document_is_one_chunk():
    chunks = split_document(doc("## 唯一小节\n很短"), max_chars=100, overlap=10)
    assert len(chunks) == 1


def test_empty_sections_are_dropped():
    chunks = split_document(doc("## 一\n\n## 二\n内容"), max_chars=100, overlap=10)
    assert [c.text for c in chunks] == ["## 一", "## 二\n内容"]


def test_long_section_is_split_into_overlapping_windows():
    text = "## 长节\n" + "甲" * 50
    chunks = split_document(doc(text), max_chars=20, overlap=5)

    assert len(chunks) > 1
    assert all(len(c.text) <= 20 for c in chunks)
    # 相邻的完整窗口之间有 overlap 个字符的重叠，关键信息不会正好断在边界上
    for before, after in zip(chunks, chunks[1:]):
        if len(before.text) == 20:
            assert before.text[-5:] == after.text[:5]


def test_real_corpus_splits_within_budget():
    for document in load_corpus():
        chunks = split_document(document)
        assert chunks, f"{document.doc_id} 没切出任何块"
        for chunk in chunks:
            assert len(chunk.text) <= 240
            assert chunk.doc_id == document.doc_id