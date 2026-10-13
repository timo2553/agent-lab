"""RAG 主链路：语料加载 -> 切分 -> 向量化 -> 检索 -> 引用回填 -> 低分兜底。"""

from agent_lab.rag.chunker import Chunk, split_document
from agent_lab.rag.corpus import CORPUS_DIR, Document, load_corpus
from agent_lab.rag.index import SearchHit, VectorIndex
from agent_lab.rag.retriever import Citation, RetrievalResult, Retriever
from agent_lab.rag.vectorizer import TfidfVectorizer, cosine_similarity, tokenize

__all__ = [
    "CORPUS_DIR",
    "Chunk",
    "Citation",
    "Document",
    "RetrievalResult",
    "Retriever",
    "SearchHit",
    "TfidfVectorizer",
    "VectorIndex",
    "build_default_retriever",
    "cosine_similarity",
    "load_corpus",
    "split_document",
    "tokenize",
]


def build_default_retriever(
    *,
    min_score: float = 0.15,
    top_k: int = 3,
    max_chars: int = 240,
    overlap: int = 40,
) -> Retriever:
    """用包内自带语料建一个检索器（CLI 与演示的默认入口）。"""
    chunks: list[Chunk] = []
    for document in load_corpus():
        chunks.extend(split_document(document, max_chars=max_chars, overlap=overlap))
    return Retriever(VectorIndex(chunks), min_score=min_score, top_k=top_k)