"""把长文档切成小块（**练习部分**）。

检索的最小单位是"块"，不是"篇"：整篇文档太长，向量会被无关内容稀释，
就算命中也定位不到具体是哪一句。所以先切。

策略是"语义优先 + 长度兜底"：
1. 先按二级标题（`## `）把文档切成若干小节，标题行跟着它的小节一起走；
   没有二级标题就把整篇当一个小节。
2. 小节没超长度上限（max_chars）就整块留下——切碎反而丢上下文。
3. 超了再按长度硬切，切成互相重叠（overlap 个字符）的窗口，
   避免关键信息正好被切断在边界上、两边都取不到。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_lab.rag.corpus import Document


@dataclass(frozen=True)
class Chunk:
    """一个可检索的文本块。chunk_id 形如 `<doc_id>#<序号>`。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str


def split_document(
    doc: Document,
    *,
    max_chars: int = 240,
    overlap: int = 40,
) -> list[Chunk]:
    """把一篇文档切成块。

    要求（测试按这个来）：
    - 以 `## ` 开头的行是一个新小节的开始；小节文本**包含**这一行标题
    - 不超 max_chars 的小节整块成一块，不切
    - 超长小节按 max_chars 切成窗口，相邻窗口重叠 overlap 个字符
    - 每个块的 chunk_id = f"{doc.doc_id}#{序号}"，序号从 0 开始、按出现顺序递增
    - 块的 doc_id / title 取自 doc；块文本两端去空白；空的小节丢弃
    - 任何块的 len(text) 都不超过 max_chars（假定 overlap < max_chars）

    返回按顺序排列的块列表。

    提示（一种可行写法）：
    1. 先把 doc.text 按行过一遍，遇到 `## ` 开头就"收束上一节、开启新一节"，
       每节用 "\\n".join(...).strip() 收尾，空的丢掉 —— 得到若干小节字符串；
    2. 对每一节：len <= max_chars 就整条成块；
       否则按步长 step = max_chars - overlap 切片 section[i : i + max_chars]，
       这样相邻窗口天然重叠 overlap 个字符（验证见测试里的重叠断言）。
    """
    
    if overlap >= max_chars:
        raise ValueError("overlap 必须小于 max_chars，否则窗口不会前进")
    sections : list[str] = []
    current: list[str] = []
    for line in doc.text.splitlines():
       if line.startswith("## "):
            section="\n".join(current).strip()
            if section:
                sections.append(section)
            current = [line]
       else:
            current.append(line)
    section = "\n".join(current).strip()
    if section:
        sections.append(section)
    chunks: list[Chunk] = []
    step = max_chars - overlap
    for section in sections:
          if len(section) <= max_chars:
             windows = [section]
          else:
             windows = [section[i : i + max_chars] for i in range(0, len(section), step)]
          for window in windows:
              if not window:
                  continue
              chunks.append(Chunk(
                 chunk_id=f"{doc.doc_id}#{len(chunks)}",
                 doc_id=doc.doc_id,
                 title=doc.title,
                 text=window,
             )
          )
    return chunks

    
