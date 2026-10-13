"""知识库语料：加载 rag/corpus/*.md 并解析出文档元数据。

这一步是"把散落的资料变成可检索的对象"。文件里把标题写成一级标题（`# `），
正文用二级标题（`## `）分小节——语料的组织结构会直接影响后面的切分和命中率。

标题行从正文里摘出来单独存成 title，正文 text 不含标题行：
这样切出来的块不会每条都重复一遍文档标题。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 语料是包内数据。声明 package-data 后（见 pyproject.toml），打 wheel 时不会丢。
CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass(frozen=True)
class Document:
    """一篇知识库文档。doc_id 取自文件名（不含扩展名）。"""

    doc_id: str
    title: str
    text: str


def load_corpus(corpus_dir: Path | str = CORPUS_DIR) -> list[Document]:
    """读取目录下所有 .md，按文件名排序返回。

    标题取正文里第一个一级标题（`# ` 开头、不是 `##`）那一行；没有就用文件名。
    正文 text 是去掉标题行之后的内容，并去掉首尾空白。
    """
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{directory}")
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = path.stem
        body_start = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                body_start = index + 1
                break
        body = "\n".join(lines[body_start:]).strip()
        documents.append(Document(doc_id=path.stem, title=title, text=body))
    return documents