"""命令行版：多文档知识库问答。

    python rag_folder.py                        # 进入交互问答
    python rag_folder.py --ask "电池能用多久"     # 只问一次
    python rag_folder.py --top-k 3 --min-score 0.45

检索与生成逻辑都在 rag_core.py，这里只负责命令行交互。
"""

from __future__ import annotations

import argparse
import time

import rag_core as rag


def show_sources(hits: list[rag.Hit]) -> None:
    print("\n检索到的片段：")
    for i, h in enumerate(hits, start=1):
        preview = h.chunk.text[:60].replace("\n", " ")
        print(f"  [{i}] {h.score:.3f}  {h.chunk.source} #{h.chunk.index + 1}  «{preview}»")


def answer_once(question: str, index, args, api_key: str | None) -> None:
    hits, refused = rag.search(question, index, top_k=args.top_k, min_score=args.min_score)
    show_sources(hits)

    if refused:
        print(f"\n最高相似度 {hits[0].score:.3f} 低于阈值 {args.min_score:.2f}，直接拒答（未调用大模型）")
        print(f"\n{rag.REFUSAL}")
        return

    print("\n思考中…")
    print(f"\n{rag.generate(question, hits, api_key)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="多文档知识库问答（命令行版）")
    ap.add_argument("--ask", help="只回答这一个问题，然后退出")
    ap.add_argument("--top-k", type=int, default=rag.DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=rag.DEFAULT_MIN_SCORE)
    ap.add_argument("--folder", default=rag.KNOWLEDGE_DIR)
    args = ap.parse_args()

    t0 = time.time()
    print(f"正在加载向量模型 {rag.EMBED_MODEL} …")
    model = rag.load_model()
    index = rag.build_index_from_folder(model, args.folder)
    lens = [len(c.text) for c in index.chunks]
    print(f"知识库就绪：{len(index.chunks)} 个片段，平均 {sum(lens) / len(lens):.0f} 字，"
          f"耗时 {time.time() - t0:.1f}s")

    api_key = rag.api_key_from_env()
    if not api_key:
        print("提示：未找到 DEEPSEEK_API_KEY，可以检索但无法生成回答。")

    if args.ask:
        answer_once(args.ask, index, args, api_key)
        return

    print("输入问题开始提问（输入 exit 退出）")
    while True:
        try:
            question = input("\n问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer_once(question, index, args, api_key)


if __name__ == "__main__":
    main()
