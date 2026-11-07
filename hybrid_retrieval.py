# -*- coding: utf-8 -*-
"""V5 检索层：向量召回 + BM25 召回 + RRF 融合。

设计原则：
- 不替换 V4 的 `rag_core.search`，而是包一层，方便开关和对比；
- 向量负责"意思像"，BM25 负责"字面像"（型号、报错码、专有名词）；
- RRF 只看排名不看分数，避免余弦和 BM25 分数量纲不一致；
- 拒答仍然看"最高余弦分"，因为 RRF 分数没有绝对阈值意义。

用法：
    from hybrid_retrieval import build_bm25, search_hybrid
    bm25 = build_bm25(index.chunks)
    hits, refused = search_hybrid("IPX7 能泡水吗", index, bm25, top_k=5, min_score=0.40)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

import rag_core as rag

_token_re = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """中文分词 + 英文/数字保留；去掉纯标点和空白。"""
    tokens: list[str] = []
    for token in jieba.lcut(text):
        token = token.strip().lower()
        if token and _token_re.fullmatch(token):
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class HybridHit:
    chunk: rag.Chunk
    rrf: float
    cosine: float
    bm25: float

    @property
    def score(self) -> float:
        """和 rag.Hit 保持同名，评测/拒答逻辑可以直接复用。"""
        return self.cosine


@dataclass
class BM25Index:
    chunks: list[rag.Chunk]
    bm25: BM25Okapi
    tokens: list[list[str]]


def build_bm25(chunks: list[rag.Chunk]) -> BM25Index:
    tokens = [tokenize(c.text) for c in chunks]
    return BM25Index(chunks=chunks, bm25=BM25Okapi(tokens), tokens=tokens)


def _vector_ranked(question: str, index: rag.Index, candidates: int) -> list[rag.Hit]:
    hits, _ = rag.search(question, index, top_k=candidates, min_score=0.0)
    return hits


def search_hybrid(
    question: str,
    index: rag.Index,
    bm25_index: BM25Index,
    top_k: int = 5,
    candidates: int = 30,
    rrf_k: int = 60,
    min_score: float = 0.0,
    vector_weight: float = 1.0,
    bm25_weight: float = 0.3,
) -> tuple[list[HybridHit], bool]:
    """向量 + BM25 双路召回，RRF 融合，返回 top_k。

    `min_score` 仍然作用在"最高余弦分"上；RRF 分数只用于排序。
    """
    vector_hits = _vector_ranked(question, index, candidates)
    cosine_by_id = {id(h.chunk): h.score for h in vector_hits}

    query_tokens = tokenize(question)
    bm25_scores = bm25_index.bm25.get_scores(query_tokens)
    bm25_order = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:candidates]
    bm25_by_id = {
        id(bm25_index.chunks[i]): float(bm25_scores[i]) for i in bm25_order
    }

    rrf: dict[int, float] = {}
    for rank, hit in enumerate(vector_hits, start=1):
        rrf[id(hit.chunk)] = rrf.get(id(hit.chunk), 0.0) + vector_weight / (rrf_k + rank)
    for rank, idx in enumerate(bm25_order, start=1):
        if bm25_scores[idx] <= 0:
            continue
        key = id(bm25_index.chunks[idx])
        rrf[key] = rrf.get(key, 0.0) + bm25_weight / (rrf_k + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    chunk_by_id = {id(c): c for c in index.chunks}
    hits = [
        HybridHit(
            chunk=chunk_by_id[key],
            rrf=score,
            cosine=cosine_by_id.get(key, 0.0),
            bm25=bm25_by_id.get(key, 0.0),
        )
        for key, score in ordered
    ]
    max_cosine = max((h.cosine for h in hits), default=0.0)
    refused = bool(min_score > 0 and (not hits or max_cosine < min_score))
    return hits, refused
