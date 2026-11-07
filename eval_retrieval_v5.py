# -*- coding: utf-8 -*-
"""RAG V5 评测：40 条正样本 + 10 条负样本。

和 V4 的 `eval_retrieval.py` 相比，这个脚本多了：
- 正样本从 15 条扩到 40 条，覆盖 4 份知识库文档；
- 指标加了 MRR@5、分文档 hit@5、正样本误拒率、拒答 precision/recall；
- 支持 `--json-out` 输出机器可读结果，方便 V5 改造前后做对比。

用法：
    python eval_retrieval_v5.py
    python eval_retrieval_v5.py --json-out eval_results/baseline_v4_50.json
    python eval_retrieval_v5.py --size 200 --overlap 40 --pack-until 90
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import hybrid_retrieval as hybrid
import rag_core as rag

# (问题, 标准答案关键词, 文档标签)
# 关键词都取自知识库原文，不依赖模型判断，结果稳定可复现。
EVAL_SET: list[tuple[str, str, str]] = [
    # ---- 产品手册（14）----
    ("电池能用多久", r"45\s*天", "产品手册"),
    ("充满电要充多久", r"4\.5\s*小时", "产品手册"),
    ("电量低了会有什么提示", r"15%|红色闪烁", "产品手册"),
    ("IPX7 防水能泡水吗", r"1\s*米水深|IPX7", "产品手册"),
    ("刷头多久需要更换", r"540\s*次|3\s*个月", "产品手册"),
    ("换完刷头怎么重置提醒", r"长按模式键\s*5\s*秒", "产品手册"),
    ("四种清洁模式有什么区别", r"31,000|38,000|26,000", "产品手册"),
    ("深层抛光模式多久用一次", r"每周\s*1[~\-]2\s*次", "产品手册"),
    ("敏感呵护模式适合什么人", r"牙龈红肿|正畸", "产品手册"),
    ("牙龈按摩模式有什么作用", r"血液循环|浮肿", "产品手册"),
    ("温度低于零度续航会变短吗", r"0℃|10%", "产品手册"),
    ("三年大概要充多少次电", r"80[~\-]90\s*次", "产品手册"),
    ("淋浴水温建议控制在多少度", r"45℃|50℃", "产品手册"),
    ("刷牙时刷头应该呈多少度", r"45°", "产品手册"),
    # ---- 常见问题（12）----
    ("刷头为什么三个月要换", r"磨圆率|菌落", "常见问题"),
    ("刷头用久了清洁效率下降多少", r"40%", "常见问题"),
    ("刷头用三个月细菌会多多少", r"15[~\-]20\s*倍", "常见问题"),
    ("机身脏了能用酒精擦吗", r"中性肥皂水|禁用酒精", "常见问题"),
    ("充电的时候进水了怎么办", r"干燥\s*24\s*小时", "常见问题"),
    ("指示灯完全不亮怎么办", r"连续充电\s*6\s*小时", "常见问题"),
    ("模式键无法切换是什么原因", r"刷头未安装到位|安全锁", "常见问题"),
    ("充电时红灯闪烁是什么原因", r"5\.5V|输入电压过高", "常见问题"),
    ("开机震动很微弱怎么办", r"充电\s*30\s*分钟", "常见问题"),
    ("怎么报修", r"二维码|顺丰|上门取件", "常见问题"),
    ("保修期是多长", r"2\s*年整机质保|保修", "常见问题"),
    ("刷头到期指示灯是什么颜色", r"黄色常亮|黄色", "常见问题"),
    # ---- 公司介绍（7）----
    ("你们公司叫什么名字", r"智净口腔科技", "公司介绍"),
    ("公司是哪一年成立的", r"2018\s*年", "公司介绍"),
    ("公司总部在哪里", r"广州", "公司介绍"),
    ("公司的电机驱动算法叫什么", r"SmartPulse", "公司介绍"),
    ("智净有哪些产品系列", r"X1|X3|X5", "公司介绍"),
    ("售后服务热线是多少", r"400-888-6688", "公司介绍"),
    ("品牌使命是什么", r"可感知的可靠", "公司介绍"),
    # ---- 推销话术（7）----
    ("这款牙刷卖多少钱", r"299|399", "推销话术"),
    ("直播间下单送什么", r"两支原装刷头|2\s*支", "推销话术"),
    ("支持几天无理由退换", r"7\s*天无理由", "推销话术"),
    ("顾客嫌贵怎么回应", r"一天不到3毛钱|2年质保", "推销话术"),
    ("顾客说普通牙刷够用怎么回应", r"巴氏刷牙法", "推销话术"),
    ("线下导购怎么开场", r"智净X3|导购", "推销话术"),
    ("能边洗澡边刷牙吗", r"淋浴|IPX7", "推销话术"),
]

# 知识库里根本没有的问题，用来验证拒答；不能和任何文档主题沾边。
NEGATIVE: list[str] = [
    "今天天气怎么样",
    "帮我写一首关于春天的诗",
    "红烧肉怎么做",
    "比特币现在什么价格",
    "智净X3的CEO叫什么名字",
    "这款牙刷有几种颜色",
    "电池用的是哪个品牌",
    "能连接手机 App 吗",
    "双十一有什么优惠",
    "公司股票代码是多少",
]


def first_match_rank(hits, pattern: str) -> int | None:
    """返回第一个命中关键词的片段排名（从 1 开始），没命中返回 None。"""
    for rank, hit in enumerate(hits, start=1):
        if re.search(pattern, hit.chunk.text):
            return rank
    return None


def make_retriever(index, mode: str, bm25_index=None,
                   vector_weight: float = 1.0, bm25_weight: float = 0.3):
    """返回一个 `(question, top_k, min_score) -> (hits, refused)` 的函数。"""
    if mode == "hybrid":
        if bm25_index is None:
            raise ValueError("hybrid 模式需要 bm25_index")
        return lambda q, top_k, min_score: hybrid.search_hybrid(
            q, index, bm25_index, top_k=top_k, min_score=min_score,
            vector_weight=vector_weight, bm25_weight=bm25_weight,
        )
    return lambda q, top_k, min_score: rag.search(
        q, index, top_k=top_k, min_score=min_score
    )


def evaluate(index, top_k: int = 5, min_score: float = 0.0, retrieve=None,
             retriever_name: str = "vector") -> dict:
    retrieve = retrieve or make_retriever(index, "vector")
    h1 = h3 = hk = 0
    rr_sum = 0.0
    misses: list[dict] = []
    pos_refused: list[str] = []
    positive_details: list[dict] = []
    per_doc: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "hit@1": 0, "hit@3": 0, "hit@5": 0, "rr": 0.0}
    )

    for q, pattern, doc in EVAL_SET:
        hits, refused = retrieve(q, max(top_k, 5), min_score)
        rank = first_match_rank(hits, pattern)
        rr = 1.0 / rank if rank else 0.0
        ok1 = rank == 1
        ok3 = rank is not None and rank <= 3
        ok5 = rank is not None and rank <= 5

        h1 += ok1
        h3 += ok3
        hk += ok5
        rr_sum += rr
        if refused:
            pos_refused.append(q)
        positive_details.append(
            {
                "question": q,
                "doc": doc,
                "rank": rank,
                "top1_score": round(hits[0].score, 4) if hits else 0.0,
                "refused": refused,
            }
        )

        bucket = per_doc[doc]
        bucket["n"] += 1
        bucket["hit@1"] += int(ok1)
        bucket["hit@3"] += int(ok3)
        bucket["hit@5"] += int(ok5)
        bucket["rr"] += rr

        if not ok5:
            misses.append(
                {
                    "question": q,
                    "doc": doc,
                    "top1_source": hits[0].chunk.source if hits else "",
                    "top1_text": hits[0].chunk.text[:40].replace("\n", " ") if hits else "",
                }
            )

    # 负样本：只看 top1 分数与是否拒答
    neg_scores: list[float] = []
    neg_refused = 0
    negative_details: list[dict] = []
    for q in NEGATIVE:
        hits, refused = retrieve(q, 3, min_score)
        score = hits[0].score if hits else 0.0
        neg_scores.append(score)
        neg_refused += int(refused)
        negative_details.append(
            {
                "question": q,
                "top1_score": round(score, 4),
                "top1_source": hits[0].chunk.source if hits else "",
                "top1_text": hits[0].chunk.text[:40].replace("\n", " ") if hits else "",
                "refused": refused,
            }
        )

    pos_scores = []
    for q, _, _ in EVAL_SET:
        hits, _ = retrieve(q, 1, 0.0)
        pos_scores.append(hits[0].score if hits else 0.0)

    true_positive = neg_refused
    false_positive = len(pos_refused)
    false_negative = len(NEGATIVE) - neg_refused
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / len(NEGATIVE) if NEGATIVE else 0.0

    for bucket in per_doc.values():
        n = bucket["n"] or 1
        bucket["hit@1"] /= n
        bucket["hit@3"] /= n
        bucket["hit@5"] /= n
        bucket["rr"] /= n

    return {
        "params": {
            "retriever": retriever_name,
            "top_k": top_k,
            "min_score": min_score,
            "chunk_size": rag.CHUNK_SIZE,
            "chunk_overlap": rag.CHUNK_OVERLAP,
            "pack_until": rag.PACK_UNTIL,
            "embed_model": rag.EMBED_MODEL,
        },
        "overall": {
            "n_positive": len(EVAL_SET),
            "n_negative": len(NEGATIVE),
            "hit@1": h1 / len(EVAL_SET),
            "hit@3": h3 / len(EVAL_SET),
            "hit@5": hk / len(EVAL_SET),
            "mrr@5": rr_sum / len(EVAL_SET),
        },
        "refusal": {
            "neg_refused": neg_refused,
            "neg_total": len(NEGATIVE),
            "pos_refused": len(pos_refused),
            "pos_total": len(EVAL_SET),
            "precision": precision,
            "recall": recall,
            "false_negative": false_negative,
        },
        "scores": {
            "pos_min": min(pos_scores),
            "pos_mean": sum(pos_scores) / len(pos_scores),
            "neg_max": max(neg_scores),
            "neg_mean": sum(neg_scores) / len(neg_scores),
        },
        "per_doc": {k: dict(v) for k, v in per_doc.items()},
        "misses": misses,
        "positive_refused": pos_refused,
        "positive_details": positive_details,
        "negative_details": negative_details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=rag.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=rag.CHUNK_OVERLAP)
    ap.add_argument("--pack-until", type=int, default=rag.PACK_UNTIL)
    ap.add_argument("--top-k", type=int, default=rag.DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=rag.DEFAULT_MIN_SCORE)
    ap.add_argument("--retriever", choices=["vector", "hybrid"], default="vector",
                    help="vector=V4 纯向量；hybrid=BM25+向量+RRF")
    ap.add_argument("--vector-weight", type=float, default=1.0,
                    help="hybrid 模式下向量一路的 RRF 权重")
    ap.add_argument("--bm25-weight", type=float, default=0.3,
                    help="hybrid 模式下 BM25 一路的 RRF 权重")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", default="", help="把完整结果写到 JSON 文件")
    args = ap.parse_args()

    chunks = rag.load_chunks(rag.KNOWLEDGE_DIR, args.size, args.overlap, args.pack_until)
    lens = [len(c.text) for c in chunks]
    print(
        f"切块: size={args.size} overlap={args.overlap} pack_until={args.pack_until}"
        f" → {len(chunks)} 段, 平均 {sum(lens) / len(lens):.0f} 字, 最长 {max(lens)}"
    )

    model = rag.load_model()
    index = rag.build_index(chunks, model)
    bm25_index = hybrid.build_bm25(chunks) if args.retriever == "hybrid" else None
    retrieve = make_retriever(
        index, args.retriever, bm25_index,
        vector_weight=args.vector_weight, bm25_weight=args.bm25_weight,
    )
    print(f"检索器: {args.retriever}")
    m = evaluate(
        index, args.top_k, args.min_score,
        retrieve=retrieve, retriever_name=args.retriever,
    )

    o = m["overall"]
    r = m["refusal"]
    s = m["scores"]
    print(
        f"检索: hit@1 = {o['hit@1']:.1%}   hit@3 = {o['hit@3']:.1%}   "
        f"hit@5 = {o['hit@5']:.1%}   MRR@5 = {o['mrr@5']:.3f}"
    )
    print(
        f"分数: 相关 top1 最低 {s['pos_min']:.3f} / 平均 {s['pos_mean']:.3f}"
        f"  |  无关 top1 最高 {s['neg_max']:.3f} / 平均 {s['neg_mean']:.3f}"
    )
    print(
        f"拒答: 阈值 {args.min_score:.2f} 时，负样本 {r['neg_refused']}/{r['neg_total']} 被拦下，"
        f"正样本误拒 {r['pos_refused']}/{r['pos_total']}；"
        f"precision={r['precision']:.2f} recall={r['recall']:.2f}"
    )
    print("分文档 hit@5:")
    for doc, b in m["per_doc"].items():
        print(f"  {doc:<6s} {b['hit@5']:.1%}  (n={b['n']})")

    if args.verbose and m["misses"]:
        print("\n未命中的问题:")
        for miss in m["misses"]:
            print(f"  [{miss['doc']}] {miss['question']:<24s} top1=[{miss['top1_text']}]")

    if args.verbose:
        print("\n负样本 top1 分数:")
        for item in sorted(m["negative_details"], key=lambda x: -x["top1_score"]):
            mark = "拒答" if item["refused"] else "漏过"
            print(
                f"  {item['top1_score']:.3f} {mark}  {item['question']:<24s}"
                f"  top1=[{item['top1_text']}]"
            )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json_out}")


if __name__ == "__main__":
    main()
