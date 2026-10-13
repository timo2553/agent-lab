"""把文本变成稀疏向量（**练习部分**）。

检索的本质是"用词衡量两段文本有多相关"。这里不引现成的向量库，
自己走一遍 TF-IDF：统计每个词在多少篇文档里出现（df），算出它的"信息量"
（idf，越少见的词越重要），再用 词频 × idf 当权重；最后用余弦相似度比较。

分词用 jieba——本仓唯一的第三方运行时依赖，取舍见 README 设计决策 11。
tokenize() 已经给你写好了；要你实现的是 fit / transform / cosine_similarity。

建议顺序：先 fit（算 idf）-> 再 transform（出向量）-> 最后 cosine_similarity，
每写完一个就跑一次对应用例：

    .\\.venv\\Scripts\\python -m pytest tests/test_retriever.py -x -q
"""

from __future__ import annotations

import logging
import math

import jieba

# jieba 首次切词会往 stderr 打"Building prefix dict..."的日志，这里压到 ERROR，
# 免得污染测试输出。
jieba.setLogLevel(logging.ERROR)


def tokenize(text: str) -> list[str]:
    """把一段文本切成词。jieba 精确模式，丢掉纯标点 / 空白，英文统一小写。"""
    return [
        token.lower()
        for token in jieba.lcut(text)
        if any(ch.isalnum() for ch in token)
    ]


class TfidfVectorizer:
    """TF-IDF 向量化器：先 fit 学 idf，再 transform 出 {词: 权重} 的稀疏向量。

    用 dict 表示稀疏向量：只存出现过的词，没出现的词隐含权重为 0。
    """

    def __init__(self) -> None:
        self.n_docs_: int = 0
        self.idf_: dict[str, float] = {}

    def fit(self, documents: list[str]) -> TfidfVectorizer:
        """统计 df，算出每个词的 idf。

        要求（测试按这个来）：
        - idf 用平滑公式：idf(t) = ln((1 + N) / (1 + df(t))) + 1，
          其中 N = 文档数，df(t) = **含该词的文档数**（同一篇里出现多次只算一次）
        - 结果存到 self.idf_（词 -> idf）；self.n_docs_ 存文档数
        - 返回 self，方便链式调用

        提示：对每篇文档用 set(tokenize(doc)) 去重后累计 df，就不会重复计数。
        """
        self.n_docs_ = len(documents)
        df_counts: dict[str, int] = {}
        for doc in documents:
            for token in set(tokenize(doc)):
                df_counts[token] = df_counts.get(token, 0) + 1
        self.idf_ = {
            token: math.log((1 + self.n_docs_) / (1 + df_counts.get(token, 0))) + 1
            for token in df_counts
        }
        return self

    def transform(self, text: str) -> dict[str, float]:
        """把一段文本变成稀疏向量 {词: 权重}。

        要求：
        - 权重 = 词频 tf × idf；tf 就用**原始出现次数**（不做归一化）
        - 不在 self.idf_ 里的词直接丢掉（训练时没见过，就没有 idf）
        - 空输入返回空 dict，不要报错
        """
        counts = self._term_frequencies(text)
        return{
            token: counts[token] * self.idf_.get(token, 0)
            for token in counts
            if token in self.idf_
        }

    def _term_frequencies(self, text: str) -> dict[str, int]:
        """可选的小工具：把 tokenize(text) 变成 {词: 出现次数}。用不用随你。"""
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        return counts


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """两个稀疏向量的余弦相似度，取值范围 [0, 1]。

    要求：
    - 分子 = 两个向量**共同词**的权重乘积之和（用 a.keys() & b.keys() 取交集）
    - 分母 = 两个向量各自模长（√各权重平方和）之积
    - 任一向量为空、或分母为 0 时返回 0.0（不能除零）

    提示：tf-idf 权重都非负，所以这里的结果天然落在 [0, 1]。
    """
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    common_keys = a_keys & b_keys
    if not common_keys:
        return 0.0
    numerator = sum(a[token] * b[token] for token in common_keys)
    denominator = math.sqrt(sum(a[token] ** 2 for token in a_keys)) * math.sqrt(sum(b[token] ** 2 for token in b_keys))
    return numerator / denominator