# -*- coding: utf-8 -*-
"""V5 实验：用 LLM 对向量 top-8 做 rerank + 可答性判断，再算指标。

这是对 `eval_retrieval_v5.py` 的补充实验，不改变 V4 默认链路：
- 检索仍是 V4 的纯向量 top-8；
- 只把"排序 + 是否可答"交给 LLM；
- 结果缓存到 `eval_results/llm_rerank_cache.json`，重复跑不重复花钱。

用法：
    python eval_llm_rerank.py
    python eval_llm_rerank.py --json-out eval_results/llm_rerank_v5_50.json
"""

from __future__ import annotations

import argparse
import json
import os

import eval_retrieval_v5 as base
import llm_rerank
import rag_core as rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=rag.DEFAULT_MIN_SCORE)
    ap.add_argument("--json-out", default="eval_results/llm_rerank_v5_50.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    api_key = rag.api_key_from_env()
    if not api_key:
        raise SystemExit("未配置 DEEPSEEK_API_KEY")

    chunks = rag.load_chunks(rag.KNOWLEDGE_DIR)
    model = rag.load_model()
    index = rag.build_index(chunks, model)

    details = []
    h1 = h3 = h5 = 0
    rr_sum = 0.0
    pos_refused = 0
    neg_refused = 0
    fallback = 0
    total_tokens = 0
    cache_hits = 0

    for q, pattern, doc in base.EVAL_SET:
        hits, _ = rag.search(q, index, top_k=args.candidates, min_score=0.0)
        out = llm_rerank.rerank(q, hits, api_key)
        if out.answerable is None:
            fallback += 1
            answerable = max((h.score for h in hits), default=0.0) >= args.min_score
        else:
            answerable = out.answerable
        ranked = out.hits[: args.top_k]
        rank = base.first_match_rank(ranked, pattern)
        rr = 1.0 / rank if rank else 0.0
        h1 += int(rank == 1)
        h3 += int(rank is not None and rank <= 3)
        h5 += int(rank is not None and rank <= 5)
        rr_sum += rr
        pos_refused += int(not answerable)
        total_tokens += int(out.usage.get("total_tokens", 0))
        cache_hits += int(out.cached)
        details.append(
            {
                "question": q,
                "doc": doc,
                "rank": rank,
                "answerable": answerable,
                "reason": out.reason,
                "cached": out.cached,
                "top1_source": ranked[0].chunk.source if ranked else "",
                "top1_text": ranked[0].chunk.text[:40].replace("\n", " ") if ranked else "",
            }
        )

    for q in base.NEGATIVE:
        hits, _ = rag.search(q, index, top_k=args.candidates, min_score=0.0)
        out = llm_rerank.rerank(q, hits, api_key)
        if out.answerable is None:
            fallback += 1
            answerable = max((h.score for h in hits), default=0.0) >= args.min_score
        else:
            answerable = out.answerable
        neg_refused += int(not answerable)
        total_tokens += int(out.usage.get("total_tokens", 0))
        cache_hits += int(out.cached)
        details.append(
            {
                "question": q,
                "doc": "负样本",
                "rank": None,
                "answerable": answerable,
                "reason": out.reason,
                "cached": out.cached,
                "top1_source": hits[0].chunk.source if hits else "",
                "top1_text": hits[0].chunk.text[:40].replace("\n", " ") if hits else "",
            }
        )

    n_pos = len(base.EVAL_SET)
    n_neg = len(base.NEGATIVE)
    tp, fp = neg_refused, pos_refused
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_neg
    result = {
        "retriever": "vector+llm_rerank",
        "params": {
            "top_k": args.top_k,
            "candidates": args.candidates,
            "min_score_fallback": args.min_score,
            "model": llm_rerank.DEFAULT_MODEL,
        },
        "overall": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "hit@1": h1 / n_pos,
            "hit@3": h3 / n_pos,
            "hit@5": h5 / n_pos,
            "mrr@5": rr_sum / n_pos,
        },
        "refusal": {
            "neg_refused": neg_refused,
            "neg_total": n_neg,
            "pos_refused": pos_refused,
            "pos_total": n_pos,
            "precision": precision,
            "recall": recall,
        },
        "cost": {
            "total_tokens": total_tokens,
            "cache_hits": cache_hits,
            "api_fallbacks": fallback,
        },
        "details": details,
    }

    print(
        f"检索+LLM rerank: hit@1 = {result['overall']['hit@1']:.1%}   "
        f"hit@3 = {result['overall']['hit@3']:.1%}   hit@5 = {result['overall']['hit@5']:.1%}   "
        f"MRR@5 = {result['overall']['mrr@5']:.3f}"
    )
    print(
        f"拒答: 负样本 {neg_refused}/{n_neg} 拦下，正样本误拒 {pos_refused}/{n_pos}；"
        f"precision={precision:.2f} recall={recall:.2f}"
    )
    print(
        f"成本: 总 token={total_tokens}，缓存命中={cache_hits}，API 失败回退={fallback}"
    )
    if args.verbose:
        print("\n明细:")
        for item in details:
            mark = "OK" if item["answerable"] else "NO"
            print(
                f"  [{mark}] [{item['doc']}] {item['question']:<24s} rank={item['rank']} "
                f"top1=[{item['top1_text']}]"
            )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json_out}")


if __name__ == "__main__":
    main()
