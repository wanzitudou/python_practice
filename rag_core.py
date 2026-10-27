"""RAG 核心逻辑：切块 → 向量化 → 检索 → 生成。

被 app.py（网页版）和 rag_folder.py（命令行版）共用，本身不含任何 UI 代码，
所以可以被单独 import、单独测试。
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# 必须在 import sentence_transformers 之前设置，否则镜像不生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------- 可调配置 ----------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
FALLBACK_MODEL = "all-MiniLM-L6-v2"          # 中文模型下载失败时的兜底

KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge_base")
# 下面三个参数由 eval_retrieval.py 在 15 条评测问题上比出来的（见 README「参数是怎么定的」）：
# 200/40/90 这组 hit@1 = 80%、hit@5 = 100%，优于 300/50 和 400/60 两组。
CHUNK_SIZE = 200                              # 每个片段的目标字数上限
CHUNK_OVERLAP = 40                            # 相邻片段重叠字数
PACK_UNTIL = 90                               # 当前片段不足这么多字时才允许并入下一块
MIN_CHUNK_LEN = 30                            # 过短的片段直接丢弃
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.40                      # 低于此分数视为"资料中没有"（见 README 的阈值校准）

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
REQUEST_TIMEOUT = 30

# bge-zh v1.5 已经不需要给 query 加检索指令；实测加了反而让相关/无关的分数间隔变小。
# 如果换成 v1 版模型，可以用环境变量 QUERY_PREFIX 打开。
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "")

REFUSAL = "哎呀，我翻遍了手册也没找到这个问题的答案，要不您换个问法试试？"


# ---------------- 数据结构 ----------------
@dataclass(frozen=True)
class Chunk:
    """一个文本片段 + 它的出处，出处是引用溯源的基础。"""
    text: str
    source: str      # 文件名
    index: int       # 文件内第几段


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Index:
    model: object
    chunks: list[Chunk]
    embeddings: object          # np.ndarray, shape (len(chunks), dim)


# ---------------- 1. 切块 ----------------
# 以序号、项目符号、Q1、小标题开头的行，都是一个"新条目"的起点
_ITEM_START = re.compile(r"^\s*(?:\d+[.、)）]|[•·\-*]\s|[Qq]\d+\s*[:：]|【|第[一二三四五六七八九十]+[章节条])")
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")


def _blocks(text: str) -> list[str]:
    """先按"段落 / 列表项 / 表格行"把文本拆成结构块，尽量不破坏表格和编号列表。"""
    blocks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            blocks.append("\n".join(buf))
            buf.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if _ITEM_START.match(line):
            flush()
            buf.append(line)
        elif "\t" in line and buf and "\t" not in buf[-1]:
            # 表格行单独成块，避免把表格和正文粘在一起
            flush()
            buf.append(line)
        else:
            buf.append(line)
    flush()
    return blocks


def _split_long(block: str, size: int) -> list[str]:
    """超长块按中文句末标点切开，再合并到接近 size 为止。"""
    if len(block) <= size:
        return [block]
    parts: list[str] = []
    cur = ""
    for sent in _SENTENCE_END.split(block):
        if not sent:
            continue
        if len(cur) + len(sent) > size and cur:
            parts.append(cur)
            cur = sent
        else:
            cur += sent
    if cur:
        parts.append(cur)
    return parts


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
               pack_until: int = PACK_UNTIL) -> list[str]:
    """把一篇文档切成片段。

    策略是"结构优先、尺寸兜底"：先按段落/编号/表格行拿到结构块，只有当当前片段
    还不足 pack_until 字时才并入下一块 —— 这样 FAQ 的一条问答不会被塞进同一段，
    而短句又不会各自成段。
    """
    pieces: list[str] = []
    for block in _blocks(text):
        pieces.extend(_split_long(block, size))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
        elif len(cur) < pack_until and len(cur) + len(piece) + 2 <= size:
            cur += "\n\n" + piece
        else:
            chunks.append(cur)
            # 用上一块尾部做重叠，避免答案正好被切在边界上
            tail = cur[-overlap:] if overlap > 0 else ""
            tail = tail[tail.find("\n") + 1:] if "\n" in tail else tail
            cur = (tail + "\n\n" + piece).strip() if tail else piece
    if cur:
        chunks.append(cur)

    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]


def load_chunks(folder: str = KNOWLEDGE_DIR,
                size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP,
                pack_until: int = PACK_UNTIL) -> list[Chunk]:
    """读取文件夹下所有 .txt，切块并保留出处。"""
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        name = os.path.basename(path)
        for i, text in enumerate(split_text(content, size, overlap, pack_until)):
            chunks.append(Chunk(text=text, source=name, index=i))
    return chunks


# ---------------- 2. 向量化 ----------------
def load_model(name: str = EMBED_MODEL):
    """加载 embedding 模型；中文模型不可用时退回英文小模型并给出提示。"""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(name)
    except Exception as exc:                      # 网络/磁盘/模型不存在
        print(f"[warn] 加载 {name} 失败（{type(exc).__name__}），回退到 {FALLBACK_MODEL}")
        return SentenceTransformer(FALLBACK_MODEL)


def build_index(chunks: Sequence[Chunk], model) -> Index:
    """把片段编码成归一化向量。归一化后，点积就等于余弦相似度。"""
    embeddings = model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return Index(model=model, chunks=list(chunks), embeddings=embeddings)


def build_index_from_folder(model, folder: str = KNOWLEDGE_DIR,
                            size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                            pack_until: int = PACK_UNTIL) -> Index:
    return build_index(load_chunks(folder, size, overlap, pack_until), model)


# ---------------- 3. 检索 ----------------
def search(question: str, index: Index, top_k: int = DEFAULT_TOP_K,
           min_score: float = 0.0) -> tuple[list[Hit], bool]:
    """返回 (命中的片段, 是否应当拒答)。

    拒答条件是"最高分低于 min_score"，这样既能省下 API 调用，也能压住幻觉。
    """
    import numpy as np

    q = index.model.encode(
        [QUERY_PREFIX + question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = np.asarray(index.embeddings) @ q
    order = np.argsort(scores)[::-1][:top_k]
    hits = [Hit(chunk=index.chunks[i], score=float(scores[i])) for i in order]
    refused = bool(min_score > 0 and (not hits or hits[0].score < min_score))
    return hits, refused


# ---------------- 4. 生成 ----------------
def build_prompt(question: str, hits: Iterable[Hit]) -> str:
    context = "\n\n".join(
        f"[{i}] 来源：{h.chunk.source}\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    )
    return f"""你是一个幽默风趣的产品推荐官，用轻松活泼的口吻回答，但事实必须来自参考资料。

要求：
1. 只依据【参考资料】作答，不要编造资料里没有的参数、价格、承诺。
2. 参考资料里确实没有答案时，就认为资料中未找到，不要猜测。
3. 用到的每条信息后面标注来源编号，例如 [1][2]。

【参考资料】
{context}

【用户问题】
{question}

回答："""


def generate(question: str, hits: Sequence[Hit], api_key: str | None) -> str:
    """调用 DeepSeek 生成最终回答。任何异常都以可读文本返回，不抛出。"""
    import requests

    if not api_key:
        return "未配置 API Key：请在 .env 里设置 DEEPSEEK_API_KEY。"
    if not hits:
        return REFUSAL

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": build_prompt(question, hits)}],
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as exc:
        return f"请求异常：{type(exc).__name__}: {exc}"

    if resp.status_code != 200:
        return f"API 请求失败（{resp.status_code}）：{resp.text[:200]}"
    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError):
        return f"返回格式异常：{resp.text[:200]}"


def api_key_from_env() -> str | None:
    """命令行版用：从环境变量（或 .env）读取 Key。"""
    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("DEEPSEEK_API_KEY")
