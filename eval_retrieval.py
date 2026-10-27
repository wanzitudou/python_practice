# -*- coding: utf-8 -*-
"""检索效果评测：用一批带标准答案关键词的问题，算 hit@1 / hit@3。

用法：
    python eval_retrieval.py
    python eval_retrieval.py --size 200 --overlap 40 --pack-until 120

这个脚本不调用 DeepSeek API，只评测"检索"这一段，所以可以廉价地反复跑，
用来对比不同切块参数、不同 embedding 模型的效果。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import rag_core as rag

# 每个问题配一个"标准答案关键词"，检索到的片段里只要能匹配上就算命中。
# 关键词都取自知识库原文，不依赖模型判断，所以结果稳定可复现。
EVAL_SET = [
    ("电池能用多久",               r"45\s*天"),
    ("刷头多久需要更换",           r"540\s*次|3\s*个月"),
    ("IPX7 防水能泡水吗",          r"1\s*米水深|IPX7"),
    ("保修期是多长",               r"2\s*年整机质保|保修"),
    ("充满电要充多久",             r"4\.5\s*小时"),
    ("电量低了会有什么提示",       r"15%|红色闪烁"),
    ("换完刷头怎么重置提醒",       r"长按模式键\s*5\s*秒"),
    ("这款牙刷卖多少钱",           r"299|399"),
    ("你们公司叫什么名字",         r"智净口腔科技"),
    ("牙龈出血用哪个模式",         r"敏感呵护"),
    ("充电的时候进水了怎么办",     r"干燥\s*24\s*小时"),
    ("能边洗澡边刷牙吗",           r"淋浴|IPX7"),
    ("刷头为什么三个月要换",       r"磨圆率|菌落"),
    ("机身脏了能用酒精擦吗",       r"中性肥皂水|禁用酒精"),
    ("线下导购怎么开场",           r"智净X3|导购"),
]

# 知识库里根本没有的问题，用来验证"能不能拒答"
NEGATIVE = [
    "今天天气怎么样",
    "帮我写一首关于春天的诗",
    "红烧肉怎么做",
    "比特币现在什么价格",
]


def hit_at(question: str, pattern: str, index, k: int) -> bool:
    hits, _ = rag.search(question, index, top_k=k)
    return any(re.search(pattern, h.chunk.text) for h in hits)


def evaluate(index, top_k: int = 5, min_score: float = 0.0) -> dict:
    h1 = h3 = hk = 0
    misses = []
    for q, pat in EVAL_SET:
        hits, _ = rag.search(q, index, top_k=top_k)
        ok1 = bool(re.search(pat, hits[0].chunk.text))
        ok3 = any(re.search(pat, h.chunk.text) for h in hits[:3])
        okk = any(re.search(pat, h.chunk.text) for h in hits)
        h1 += ok1
        h3 += ok3
        hk += okk
        if not okk:
            misses.append((q, hits[0].chunk.text[:36]))

    # 负样本：top1 分数越低越好
    neg_scores = [rag.search(q, index, top_k=1)[0][0].score for q in NEGATIVE]
    pos_scores = [rag.search(q, index, top_k=1)[0][0].score for q, _ in EVAL_SET]

    refused = sum(1 for q in NEGATIVE if rag.search(q, index, top_k=3, min_score=min_score)[1])
    return {
        "hit@1": h1 / len(EVAL_SET),
        "hit@3": h3 / len(EVAL_SET),
        f"hit@{top_k}": hk / len(EVAL_SET),
        "misses": misses,
        "pos_min": min(pos_scores),
        "pos_mean": sum(pos_scores) / len(pos_scores),
        "neg_max": max(neg_scores),
        "neg_mean": sum(neg_scores) / len(neg_scores),
        "neg_refused": f"{refused}/{len(NEGATIVE)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=rag.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=rag.CHUNK_OVERLAP)
    ap.add_argument("--pack-until", type=int, default=rag.PACK_UNTIL)
    ap.add_argument("--top-k", type=int, default=rag.DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=rag.DEFAULT_MIN_SCORE)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    chunks = rag.load_chunks(rag.KNOWLEDGE_DIR, args.size, args.overlap, args.pack_until)
    lens = [len(c.text) for c in chunks]
    print(f"切块: size={args.size} overlap={args.overlap} pack_until={args.pack_until}"
          f" → {len(chunks)} 段, 平均 {sum(lens) / len(lens):.0f} 字, 最长 {max(lens)}")

    model = rag.load_model()
    index = rag.build_index(chunks, model)

    m = evaluate(index, args.top_k, args.min_score)
    print(f"检索: hit@1 = {m['hit@1']:.1%}   hit@3 = {m['hit@3']:.1%}   hit@{args.top_k} = {m[f'hit@{args.top_k}']:.1%}")
    print(f"分数: 相关 top1 最低 {m['pos_min']:.3f} / 平均 {m['pos_mean']:.3f}"
          f"  |  无关 top1 最高 {m['neg_max']:.3f} / 平均 {m['neg_mean']:.3f}")
    print(f"拒答: 阈值 {args.min_score:.2f} 时，{m['neg_refused']} 个无关问题被拦下")

    if args.verbose and m["misses"]:
        print("\n未命中的问题:")
        for q, top in m["misses"]:
            print(f"  {q:<22s} top1=«{top}»")


if __name__ == "__main__":
    main()
