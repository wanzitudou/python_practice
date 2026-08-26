import streamlit as st
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import numpy as np
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ---------- 1. 环境配置 ----------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("❌ 未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")

# ---------- 2. 加载模型（缓存） ----------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
# 从网上下载一个能把中文变成数字指纹的 AI 小模型
model = load_model()

# ---------- 3. 读取知识库（绝对路径） ----------
@st.cache_data
def load_knowledge_base():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "knowledge_base")
    
    all_chunks = []
    file_paths = glob.glob(os.path.join(folder, "*.txt"))
    
    print("📁 知识库路径:", folder)
    print("📄 找到的 .txt 文件:", file_paths)
    
    if not file_paths:
        st.warning(f"⚠️ 在 {folder} 下未找到任何 .txt 文件，请检查文件夹是否存在。")
        return []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # ---- 新的切分逻辑 ----
            lines = content.split('\n')
            current_chunk = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 如果遇到数字序号（如 "1."、"2."）或表格行（包含 "|"），并且当前块已有内容，则保存当前块并开始新块
                if re.match(r'^(\d+\.|\•|\-|\|)', line) and current_chunk:
                    all_chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                current_chunk.append(line)
            # 保存最后一个块
            if current_chunk:
                all_chunks.append('\n'.join(current_chunk))
    
    # 过滤掉过短的块（少于10个字符）
    all_chunks = [chunk for chunk in all_chunks if len(chunk) > 10]
    print("📚 总共切分出", len(all_chunks), "个文本块")
    return all_chunks

chunks = load_knowledge_base()

if chunks:
    # ---------- 4. 向量化所有文本块（缓存） ----------
    @st.cache_data
    def embed_chunks(chunks):
        return model.encode(chunks, show_progress_bar=False)
    
    chunk_embeddings = embed_chunks(chunks)
else:
    chunk_embeddings = None

# ---------- 5. 核心检索 + 问答函数 ----------
def ask_question(question):
    if not chunks or chunk_embeddings is None:
        return "知识库为空，请检查 knowledge_base 文件夹是否包含 .txt 文件。"
    
    # 向量化问题
    q_emb = model.encode([question])[0]
    
    # 余弦相似度
    similarities = np.dot(chunk_embeddings, q_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    
    # 取前 5 个最相关段落
    top_indices = np.argsort(similarities)[-10:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    
    # 构造 Prompt（强约束版）
    prompt = f"""
现在你是一个幽默风趣的产品推荐官，要用轻松活泼的口吻回答。
请根据以下【参考资料】回答用户的问题。如果参考资料中有相关信息，请直接引用或概括回答。如果完全没有，请只回答“资料中未找到”。

【参考资料】
{context}

【用户问题】
{question}

请回答：
"""
    # 调用 API
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API 请求失败: {response.status_code} - {response.text}"
    except Exception as e:
        return f"请求异常: {str(e)}"

# ---------- 6. Streamlit 界面 ----------
st.set_page_config(page_title="多文档智能问答", page_icon="📚")
st.title("📚 多文档智能问答助手")
st.write("基于 `knowledge_base` 文件夹中的文档回答问题（支持多文件）")

user_question = st.text_input("请输入你的问题：")

if st.button("提交问题"):
    if user_question:
        with st.spinner("🔍 正在检索并生成回答..."):
            answer = ask_question(user_question)
        st.success("回答：")
        st.write(answer)
    else:
        st.warning("请先输入一个问题！")