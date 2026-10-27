import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 强制走国内镜像

import glob
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
# ... 其余代码不变

# ---------- 0. 全局配置 ----------
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- 1. 仓库管理员：读取并切分文档 ----------
def load_docs(folder="knowledge_base"):
    """从文件夹读取所有 txt，按段落/标点切分成短句列表"""
    all_chunks = []
    for file_path in glob.glob(os.path.join(folder, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
           # 先按序号（1. / 2. / 等）分割，再按句号分割
            import re
            items = re.split(r'\n(?=\d+\.)', content)  # 按“换行+数字序号”分割
            for item in items:
                item = item.strip()
                if len(item) > 20:
                    all_chunks.append(item)
    print(f"📚 共切分出 {len(all_chunks)} 个文本块")
    return all_chunks

# ---------- 2. 指纹录入员：把文本块全变成向量 ----------
def build_index(chunks):
    """把所有文本块转成向量（指纹），并和原文一一对应"""
    embeddings = MODEL.encode(chunks, show_progress_bar=True)
    print(f"✅ 向量化完成，共 {len(embeddings)} 个向量")
    return embeddings  # 返回一个二维数组

# ---------- 3. 接待员：根据问题找出最相似的原文 ----------
def retrieve(question, chunks, embeddings, top_k=10):
    """接收问题，返回最相似的 top_k 个原文片段"""
    q_emb = MODEL.encode([question])[0]
    # 计算相似度
    similarities = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    return context

# ---------- 4. 秘书：把原文和问题发给大模型 ----------
def generate(question, context):
    """基于检索到的原文，调用 DeepSeek 生成最终回答"""
    if not API_KEY:
        return "❌ 未配置 API Key"

    prompt = f"""
现在你是一个幽默风趣的产品推荐官，要用轻松活泼的口吻回答。
请根据以下【参考资料】回答用户问题。如果资料中没有相关信息，请用轻松的口吻说：“哎呀，我翻遍了手册也没找到这个问题的答案，要不您换个问法试试？”
【参考资料】
{context}

【用户问题】
{question}

回答：
"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"API 错误: {resp.status_code}"
    except Exception as e:
        return f"请求异常: {str(e)}"

# ---------- 5. 主程序入口（把流水线串起来） ----------
if __name__ == "__main__":
    print("🔍 正在启动多文档问答系统...")
    
    # 1. 准备原材料
    text_chunks = load_docs("knowledge_base")
    if not text_chunks:
        print("❌ 知识库为空，请检查 knowledge_base 文件夹")
        exit()

    # 2. 建立指纹库
    index = build_index(text_chunks)

    # 3. 进入问答循环
    while True:
        q = input("\n请输入你的问题（输入 exit 退出）: ")
        if q.lower() == "exit":
            break
        if not q.strip():
            continue

        print("🤔 思考中...")
        
        # 4. 根据问题找原文（接待员）
        found_context = retrieve(q, text_chunks, index, top_k=10)
        
        # 5. 把原文和问题交给大模型（秘书）
        answer = generate(q, found_context)
        
        print(f"\n🤖 {answer}")