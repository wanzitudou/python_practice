import os
import re
import glob
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 强制从当前脚本所在目录加载 .env 文件
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")   # 或者 "knowledge_base.env"
load_dotenv(dotenv_path=env_path)

# ---------- 1. 加载向量化模型（将文字转为数字指纹） ----------
print("⏳ 正在加载 AI 语义模型（首次运行会下载约 90MB）...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级且高效
print("✅ 模型加载完成！")

# ---------- 2. 读取并切分文件夹里的所有文本 ----------
def load_documents(folder_path):
    import re
    all_chunks = []
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not file_paths:
        print(f"❌ 在 {folder_path} 下没有找到 .txt 文件")
        return []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()   # 必须在此定义
            # 按中文标点切分
            sentences = re.split(r'[。！？；\n]+', content)
            for s in sentences:
                s = s.strip()
                if len(s) > 20:
                    all_chunks.append(s)
    
    print(f"📚 共读取 {len(file_paths)} 个文件，切分为 {len(all_chunks)} 个文本块")
    return all_chunks

# 指定你的知识库文件夹（因为 rag_folder.py 在根目录，所以指向子文件夹）
folder = "knowledge_base"
chunks = load_documents(folder)

if not chunks:
    exit()

# ---------- 3. 把所有文本块转换成向量（存在内存中） ----------
print("⏳ 正在将文本块转换为向量...")
chunk_embeddings = model.encode(chunks, show_progress_bar=True)
print(f"✅ 向量化完成，共 {len(chunk_embeddings)} 个向量，每个维度 384")

# ---------- 4. 核心检索函数 ----------
def ask_question(question):
    # 4.1 将问题转为向量
    q_emb = model.encode([question])[0]
    
    # 4.2 计算问题向量与所有文本块向量的余弦相似度
    similarities = np.dot(chunk_embeddings, q_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    
    # 4.3 取出相似度最高的前 5个文本块
    top_indices = np.argsort(similarities)[-5:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    
    # ----- 调试输出（查看检索结果）-----
    print("\n" + "="*40)
    print("检索到的前 3 个文本块：")
    for idx, i in enumerate(top_indices):
        print(f"\n--- 第 {idx+1} 块 (相似度: {similarities[i]:.4f}) ---")
        print(chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i])
    print("="*40 + "\n")
    # ----- 调试结束 -----

    # 4.4 构造提示词（注意缩进）
    prompt = f"""
请根据【参考资料】回答用户问题。

【要求】
- 如果问题问的是“产品名称”、“品牌”、“功能”、“优点”，请优先从参考资料中提取明确的词汇或短语回答。
- 如果参考资料中没有直接提及，请回答“资料中未找到”。

【参考资料】
{context}

【用户问题】
{question}

请回答：
"""

    # 4.5 获取 API Key（已在顶部加载环境变量，可直接读取）
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "❌ 错误：未找到 DEEPSEEK_API_KEY，请检查 .env 文件配置。"

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-v4-flash",   # 或者 deepseek-chat
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API 请求失败: {response.status_code} - {response.text}"
    except Exception as e:
        return f"请求异常: {str(e)}"

# ---------- 5. 交互式命令行测试 ----------
if __name__ == "__main__":
    print("\n🔍 知识库问答已启动！输入 'exit' 退出。")
    while True:
        q = input("\n请输入你的问题: ")
        if q.lower() == 'exit':
            break
        if not q.strip():
            continue
        print("🤔 思考中...")
        answer = ask_question(q)
        print(f"\n🤖 回答: {answer}")