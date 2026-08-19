# 📚 多文档智能问答系统（RAG 实现）

> 一个基于**检索增强生成（RAG）**技术的本地知识库问答系统。支持从文件夹中批量读取文档，语义检索最相关的段落，并调用大模型生成精准回答。

## 🚀 功能特点

- ✅ **多文档支持**：自动读取 `knowledge_base/` 文件夹下的所有 `.txt` 文档
- ✅ **语义检索**：使用 `sentence-transformers` 将文本向量化，通过余弦相似度匹配最相关的段落
- ✅ **精准回答**：基于检索到的资料调用 DeepSeek API 生成回答，有效减少“幻觉”
- ✅ **交互方式**：提供**命令行版本**（`rag_folder.py`）和**网页版本**（`app.py`）双端

## 📁 项目结构

```
.
├── app.py                  # Streamlit 网页版（单文档问答）
├── rag_folder.py           # 命令行版（多文档 RAG）⭐ 核心
├── knowledge_base/         # 知识库文件夹（可自由添加 .txt 文档）
│   ├── 产品手册.txt
│   ├── 公司介绍.txt
│   └── 常见问题.txt
├── requirements.txt        # 项目依赖
└── .env                    # 环境变量（需自行创建，不上传）
```

## ⚙️ 技术栈

- **向量化模型**：`sentence-transformers/all-MiniLM-L6-v2`（本地运行，无需联网）
- **大模型 API**：DeepSeek API
- **Web 框架**：Streamlit
- **核心库**：`numpy`, `requests`, `python-dotenv`

## 🔧 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/wanzitudou/python_practice.git
cd python_practice

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key（创建 .env 文件）
DEEPSEEK_API_KEY=sk-你的真实Key

# 4. 运行命令行版（多文档 RAG）
python rag_folder.py

# 5. 运行网页版（单文档问答）
streamlit run app.py
```

## 🧪 运行效果示例

**用户提问**：“公司介绍”

**检索到的相关段落**：
> 公司背景： 智净口腔科技是一家专注于声波电动牙刷...核心团队来自消费电子、电机控制和口腔医学领域...

**AI 回答**：
> 根据参考资料，智净口腔科技是一家专注于声波电动牙刷及口腔健康管理系统的创新型科技企业。核心团队来自消费电子、电机控制和口腔医学领域，拥有超过10年的行业经验...

## 📝 版本说明

- **V2（`rag_folder.py`）**：多文档 RAG 系统，支持向量检索和本地语义匹配，是项目的核心。
- **V1（`app.py`）**：早期版本，基于单文档的简单问答，保留作为功能对比和网页展示。

## 🧠 学习心得

本项目是我学习 RAG 技术过程中的实践产出。从最初单文档调用 API，到实现完整的多文档向量检索与生成链路，我深入理解了文档切分、向量化、相似度检索等关键技术。

## 👤 作者
Meatball Potatoes

