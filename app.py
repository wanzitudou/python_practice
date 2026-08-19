import streamlit as st
import os
import glob
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. 加载环境变量（API Key）
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("❌ 未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")

# 2. 加载向量化模型（首次运行会下载）
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 3. 读取知识库文件夹（自动扫描 knowledge_base 下的所有 txt）
@st.cache_data
def load_knowledge_base(folder="knowledge_base"):
    all_chunks = []
    file_paths = glob.glob(os.path.join(folder, "*.txt"))
    if not file_paths:
        return []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 按中文标点切分（与你的 V2 保持一致）
            import re
            sentences = re.split(r'[。！？；\n]+', content)
            for s in sentences:
                s = s.strip()
                if len(s) > 20:
                    all_chunks.append(s)
    return all_chunks

chunks = load_knowledge_base()

if not chunks:
    st.warning("⚠️ 未在 knowledge_base 文件夹中找到 .txt 文件")

# 4. 向量化所有文本块（只在加载时计算一次）
@st.cache_data
def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=False)

if chunks:
    chunk_embeddings = embed_chunks(chunks)

# 5. 核心检索 + 问答函数（完全照搬 V2 的逻辑）
def ask_question(question):
    if not chunks:
        return "知识库为空，请检查 knowledge_base 文件夹。"
    
    # 向量化问题
    q_emb = model.encode([question])[0]
    
    # 余弦相似度计算
    similarities = np.dot(chunk_embeddings, q_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    
    # 取前 3 个最相关的段落
    top_indices = np.argsort(similarities)[-3:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    
    # 构造 Prompt（强硬约束版）
    prompt = f"""
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

# 6. Streamlit 网页界面
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