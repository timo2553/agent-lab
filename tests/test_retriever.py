"""检索的目标行为：TF-IDF 排序、top_k、低分判定、引用能对上索引。

向量化的三处核心算法（idf / tf-idf / 余弦）由你实现；这里先把"应该长什么样"钉住。
"""

from __future__ import annotations

from agent_lab.rag import build_default_retriever
from agent_lab.rag.chunker import split_document
from agent_lab.rag.corpus import Document
from agent_lab.rag.index import VectorIndex
from agent_lab.rag.retriever import Retriever
from agent_lab.rag.vectorizer import TfidfVectorizer, cosine_similarity, tokenize

DOCS = [
    Document(
        doc_id="shipping",
        title="发货与物流",
        text="## 发货时间\n现货商品付款后 24 小时内发货\n## 物流查询\n发货后可以查看物流单号，也可以找客服代查",
    ),
    Document(
        doc_id="returns",
        title="退换货政策",
        text="## 七天无理由退货\n签收后 7 天内可以申请无理由退货\n## 退款\n退货验收后发起退款",
    ),
    Document(
        doc_id="faq",
        title="商品常见问题",
        text="## 尺码\n可以把身高体重发给客服推荐尺码\n## 支付\n支持微信支付和支付宝",
    ),
]


def build_index() -> VectorIndex:
    chunks = []
    for document in DOCS:
        chunks.extend(split_document(document, max_chars=200, overlap=20))
    return VectorIndex(chunks)


# ---------- 向量化（练习部分） ----------

def test_tokenize_keeps_words_and_drops_punctuation():
    tokens = tokenize("现货商品，付款后 24 小时内发货！")
    assert "现货" in tokens and "发货" in tokens
    assert "，" not in tokens and "！" not in tokens


def test_idf_is_higher_for_rarer_terms():
    vectorizer = TfidfVectorizer().fit([d.text for d in DOCS])
    assert vectorizer.n_docs_ == 3
    # "发货"只在一篇里出现，"客服"在两篇里出现 -> 前者更有区分度、idf 更大
    assert vectorizer.idf_["发货"] > vectorizer.idf_["客服"]


def test_transform_is_tf_times_idf_and_drops_unknown_terms():
    vectorizer = TfidfVectorizer().fit([d.text for d in DOCS])
    vector = vectorizer.transform("发货 发货 客服")
    assert vector["发货"] == 2 * vectorizer.idf_["发货"]  # tf = 2
    assert vector["客服"] == 1 * vectorizer.idf_["客服"]
    assert vectorizer.transform("量子纠缠黑洞") == {}  # 训练没见过的词全丢
    assert vectorizer.transform("") == {}


def test_cosine_similarity_bounds():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0
    assert cosine_similarity({"a": 1.0}, {"a": 2.0}) == 1.0  # 同方向
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0  # 无共同词


# ---------- 索引与排序 ----------

def test_relevant_document_ranks_first():
    hits = build_index().search("多久发货", top_k=3)
    assert hits[0].chunk.doc_id == "shipping"
    assert hits[0].score > 0


def test_scores_are_sorted_desc():
    scores = [hit.score for hit in build_index().search("退货 退款 运费", top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results():
    index = build_index()
    assert len(index.search("发货", top_k=1)) == 1
    assert len(index.search("发货", top_k=100)) == len(index.chunks)


def test_unrelated_query_scores_zero():
    hits = build_index().search("量子纠缠黑洞", top_k=3)
    assert all(hit.score == 0.0 for hit in hits)


# ---------- Retriever：阈值与引用 ----------

def test_low_score_is_not_confident():
    result = Retriever(build_index(), min_score=0.15).retrieve("量子纠缠黑洞")
    assert result.confident is False
    assert result.citations == ()


def test_confident_result_has_numbered_citations_matching_the_index():
    index = build_index()
    result = Retriever(index, min_score=0.15).retrieve("多久发货")

    assert result.confident is True
    assert result.citations[0].marker == 1
    by_id = {chunk.chunk_id: chunk.text for chunk in index.chunks}
    for citation in result.citations:
        assert citation.chunk_id in by_id
        assert citation.text == by_id[citation.chunk_id]  # 引用对上真实片段


def test_context_and_sources_formatting():
    result = Retriever(build_index(), min_score=0.15).retrieve("多久发货")
    context = Retriever.format_context(result.citations)

    assert "【1】" in context
    assert result.citations[0].text in context
    assert Retriever.format_sources(result.citations).startswith("[1]")


# ---------- 真实语料 ----------

def test_real_corpus_is_searchable():
    result = build_default_retriever().retrieve("退款多久到账")
    assert result.confident is True
    assert any(citation.doc_id == "returns" for citation in result.citations)