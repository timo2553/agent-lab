"""检索器：在向量索引之上加"够不够可信"的判断，并把命中整理成带编号的引用。

低分兜底就发生在这里：最高分都低于阈值，就认为"知识库里没有"，返回
confident=False，让上层直接拒答、不调主模型。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_lab.rag.index import SearchHit, VectorIndex


@dataclass(frozen=True)
class Citation:
    """一条可回填到回答里的引用：marker 是回答里用的编号 [1][2]。"""

    marker: int
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的结果。confident=False 表示低分、应当兜底。"""

    query: str
    hits: tuple[SearchHit, ...]
    citations: tuple[Citation, ...]
    confident: bool


class Retriever:
    """索引 + 信誉阈值：分数不够就不放行。"""

    def __init__(
        self,
        index: VectorIndex,
        *,
        min_score: float = 0.15,
        top_k: int = 3,
    ) -> None:
        self.index = index
        self.min_score = min_score
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        hits = tuple(self.index.search(query, top_k=self.top_k))
        confident = bool(hits) and hits[0].score >= self.min_score
        citations = (
            tuple(
                Citation(
                    marker=position + 1,
                    chunk_id=hit.chunk.chunk_id,
                    doc_id=hit.chunk.doc_id,
                    title=hit.chunk.title,
                    text=hit.chunk.text,
                    score=hit.score,
                )
                for position, hit in enumerate(hits)
            )
            if confident
            else ()
        )
        return RetrievalResult(
            query=query, hits=hits, citations=citations, confident=confident
        )

    @staticmethod
    def format_context(citations: tuple[Citation, ...]) -> str:
        """把引用拼成注入 prompt 的"参考资料"，每条带可回填的编号。"""
        return "\n\n".join(
            f"【{c.marker}】《{c.title}》\n{c.text}" for c in citations
        )

    @staticmethod
    def format_sources(citations: tuple[Citation, ...]) -> str:
        """把引用拼成给用户看的来源标注，例如：[1]《发货与物流时效》。"""
        return "、".join(f"[{c.marker}]《{c.title}》" for c in citations)