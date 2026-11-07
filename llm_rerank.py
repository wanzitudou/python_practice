# -*- coding: utf-8 -*-
"""V5 可选：用大模型做 rerank + 可答性判断。

为什么不用 cross-encoder：
- 本机下载 bge-reranker 权重时网络卡住（0 字节），先做一个可运行的 LLM 版本；
- 它同时输出 `answerable`，可以补上纯阈值拒答在负样本上的短板；
- 代价是延迟和费用，所以默认关闭，只作为评测/离线实验。

设计：
- 输入 top-N 候选（默认 10），让模型只输出 JSON：{"answerable": bool, "ranking": [..]}；
- 结果按 (问题 + 候选片段 hash + 模型) 缓存，重复评测不重复花钱；
- API 失败返回 answerable=None，由调用方退回阈值拒答，不阻塞主链路。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import requests

import rag_core as rag

DEFAULT_MODEL = os.getenv("RERANK_MODEL", "deepseek-chat")
CACHE_PATH = os.getenv("RERANK_CACHE", "eval_results/llm_rerank_cache.json")
MAX_CHUNK_CHARS = 280


@dataclass
class RerankOutcome:
    hits: list[rag.Hit]
    answerable: bool | None
    reason: str
    usage: dict
    cached: bool


def _cache_key(question: str, hits: list[rag.Hit]) -> str:
    raw = question + "\n" + "\n".join(h.chunk.text for h in hits)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(path: str, cache: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _build_prompt(question: str, hits: list[rag.Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        text = hit.chunk.text[:MAX_CHUNK_CHARS].replace("\n", " ")
        lines.append(f"[{i}] 来源：{hit.chunk.source}\n{text}")
    return f"""你是 RAG 检索结果审核器。不要回答用户问题，只做两件事：
1. 判断下面这些资料里是否有能直接回答问题的信息（answerable）；
2. 如果有，把最相关的 3 条按相关性从高到低排序（ranking 写资料编号）。

用户问题：{question}

候选资料：
{chr(10).join(lines)}

只输出 JSON，格式：
{{"answerable": true, "ranking": [1, 2, 3], "reason": "一句话理由"}}
"""


def rerank(
    question: str,
    hits: list[rag.Hit],
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    cache_path: str = CACHE_PATH,
    timeout: int = 30,
) -> RerankOutcome:
    """对候选片段重排，并判断资料是否足以回答。失败时 answerable=None。"""
    if not hits:
        return RerankOutcome([], False, "no candidates", {}, False)
    if not api_key:
        return RerankOutcome(hits, None, "no api key", {}, False)

    cache = _load_cache(cache_path)
    key = _cache_key(question, hits)
    if key in cache:
        item = cache[key]
        return RerankOutcome(
            _apply_ranking(hits, item.get("ranking", [])),
            item.get("answerable"),
            item.get("reason", ""),
            item.get("usage", {}),
            True,
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_prompt(question, hits)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            rag.DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:
        return RerankOutcome(hits, None, f"request error: {type(exc).__name__}", {}, False)

    if resp.status_code != 200:
        return RerankOutcome(hits, None, f"http {resp.status_code}", {}, False)

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        ranking = [int(x) for x in data.get("ranking", []) if 1 <= int(x) <= len(hits)]
        answerable = bool(data.get("answerable"))
        reason = str(data.get("reason", ""))[:120]
        usage = resp.json().get("usage", {})
    except (KeyError, IndexError, ValueError, TypeError):
        return RerankOutcome(hits, None, "invalid json", {}, False)

    cache[key] = {
        "answerable": answerable,
        "ranking": ranking,
        "reason": reason,
        "usage": usage,
    }
    _save_cache(cache_path, cache)
    return RerankOutcome(_apply_ranking(hits, ranking), answerable, reason, usage, False)


def _apply_ranking(hits: list[rag.Hit], ranking: list[int]) -> list[rag.Hit]:
    seen: set[int] = set()
    ordered: list[rag.Hit] = []
    for idx in ranking:
        if 1 <= idx <= len(hits) and idx not in seen:
            ordered.append(hits[idx - 1])
            seen.add(idx)
    ordered.extend(hit for i, hit in enumerate(hits, start=1) if i not in seen)
    return ordered
