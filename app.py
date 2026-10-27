"""Streamlit 网页版：多文档知识库问答（带引用溯源与低分拒答）。

运行：streamlit run app.py
UI 只管展示，检索与生成逻辑都在 rag_core.py 里。
"""

from __future__ import annotations

import glob
import os

import streamlit as st

import rag_core as rag

st.set_page_config(page_title="多文档智能问答", page_icon="📚", layout="centered")


# ---------------- 缓存 ----------------
@st.cache_resource(show_spinner=False)
def get_index(size: int, overlap: int, pack_until: int) -> rag.Index:
    """模型 + 切块 + 向量矩阵一起缓存：这是一次性重活，只做一次。"""
    model = rag.load_model()
    return rag.build_index_from_folder(model, rag.KNOWLEDGE_DIR, size, overlap, pack_until)


def get_api_key() -> str | None:
    """优先 Streamlit Secrets（部署用），退回 .env / 环境变量（本地用）。"""
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("DEEPSEEK_API_KEY")


# ---------------- 页面（先渲染快的部分）----------------
st.title("多文档智能问答")
st.caption("答案只来自 `knowledge_base/` 里的资料，并标注引用来源；相似度太低时直接拒答，不调用大模型。")

with st.sidebar:
    st.subheader("检索设置")
    top_k = st.slider("召回片段数", 1, 10, rag.DEFAULT_TOP_K,
                      help="送给大模型参考的片段数量，越多越不容易漏，但上下文更长。")
    min_score = st.slider("相似度阈值", 0.0, 1.0, rag.DEFAULT_MIN_SCORE, 0.01,
                          help="最高相似度低于该值时直接拒答，避免把无关内容喂给模型。")
    show_sources = st.toggle("显示检索来源", value=True)
    st.divider()
    st.caption(f"向量模型：`{rag.EMBED_MODEL}`")
    st.caption(f"切块：{rag.CHUNK_SIZE} 字 / 重叠 {rag.CHUNK_OVERLAP} 字")

    with st.expander("知识库文件"):
        for path in sorted(glob.glob(os.path.join(rag.KNOWLEDGE_DIR, "*.txt"))):
            st.caption(f"· {os.path.basename(path)}")

    if not get_api_key():
        st.warning("未检测到 DEEPSEEK_API_KEY，将无法生成回答。", icon=":material/key_off:")

with st.form("ask", border=True):
    question = st.text_input("你的问题", placeholder="例如：电池能用多久？")
    submitted = st.form_submit_button("提问", icon=":material/search:", type="primary")

st.caption("可以试试：电池能用多久 / 刷头多久换一次 / IPX7 防水能泡水吗 / 保修期是多长")

answer_slot = st.container()

# ---------------- 慢活放在最后 ----------------
if submitted:
    q = question.strip()
    if not q:
        answer_slot.warning("请先输入一个问题。", icon=":material/edit:")
    else:
        with st.spinner("正在检索…首次运行需要加载向量模型，大约 20 秒。"):
            index = get_index(rag.CHUNK_SIZE, rag.CHUNK_OVERLAP, rag.PACK_UNTIL)
            hits, refused = rag.search(q, index, top_k=top_k, min_score=min_score)

        if refused:
            with answer_slot.container(border=True):
                st.markdown(rag.REFUSAL)
            st.caption(
                f"最高相似度 {hits[0].score:.3f} 低于阈值 {min_score:.2f}，"
                f"已跳过模型调用（省下一次 API 费用）。"
            )
        else:
            with st.spinner("正在生成回答…"):
                answer = rag.generate(q, hits, get_api_key())
            with answer_slot.container(border=True):
                st.markdown(answer)

        if show_sources:
            st.subheader("检索来源")
            st.dataframe(
                [
                    {
                        "出处": f"{h.chunk.source} #{h.chunk.index + 1}",
                        "相似度": round(h.score, 3),
                        "片段": h.chunk.text[:70].replace("\n", " "),
                    }
                    for h in hits
                ],
                column_config={
                    "相似度": st.column_config.ProgressColumn(
                        "相似度", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "出处": st.column_config.TextColumn("出处", width="small"),
                },
                hide_index=True,
            )
