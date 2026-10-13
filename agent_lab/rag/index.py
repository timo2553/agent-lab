"""把块建成向量索引，并支持按 query 检索。

这里是"编排"，不是练习部分：它调用 vectorizer 把每个块编成向量，检索时
把 query 也编成向量，逐个算余弦相似度，按分数从高到低排。

核心算法（idf、tf-idf 权重、余弦相似度）在 vectorizer.py 里，由你实现。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_lab.rag.chunker import Chunk
from agent_lab.rag.vectorizer import TfidfVectorizer, cosine_similarity


@dataclass(frozen=True)
class SearchHit:
    """一次命中：命中的块 + 相似度分数。"""

    chunk: Chunk
    score: float


class VectorIndex:
    """基于 TF-IDF 的向量索引：构造时一次性把块编好，检索时只编 query。"""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        vectorizer: TfidfVectorizer | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.vectorizer = vectorizer or TfidfVectorizer()
        # 用全部块训练 idf，再把每块编成向量缓存起来（检索时不必重复算）
        self.vectorizer.fit([chunk.text for chunk in self.chunks])
        self._vectors = [self.vectorizer.transform(chunk.text) for chunk in self.chunks]

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """返回相似度最高的 top_k 个块，按分数降序。"""
        query_vector = self.vectorizer.transform(query)
        hits = [
            SearchHit(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self._vectors)
        ]
        # 稳定排序：分数相同时保持块的原始顺序，便于复现
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]