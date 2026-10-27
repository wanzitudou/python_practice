# 📚 多文档智能问答系统（RAG 实现）

> 一个**不依赖任何 RAG 框架、手写检索内核**的本地知识库问答系统。自动扫描 `knowledge_base/` 下的多份文档，用向量检索找出最相关的段落，再调用大模型生成有据可依的回答。

**核心特点：向量化、相似度计算、top-k 召回全部自己实现**（`sentence-transformers` 编码 + `numpy` 余弦相似度），没有使用 LangChain / FAISS / Chroma，目的是把 RAG 的每个环节都亲手写一遍。

---

## ✨ 功能特点

- ✅ **多文档知识库**：自动扫描 `knowledge_base/` 下所有 `.txt`，加文档不需要改代码
- ✅ **手写语义检索**：`all-MiniLM-L6-v2` 把文本编码成 384 维向量，用余弦相似度召回 top-10 片段
- ✅ **受约束生成**：把检索到的原文作为【参考资料】注入 Prompt，要求"资料中没有就说没找到"，抑制幻觉
- ✅ **双入口**：Streamlit 网页版（`app.py`）+ 命令行版（`rag_folder.py`）
- ✅ **工程细节**：模型与向量索引双层缓存、`__file__` 锚定路径、HuggingFace 国内镜像、API 异常兜底

---

## 📁 项目结构

```
.
├── app.py                            # ⭐ Streamlit 网页版（当前主力）
├── rag_folder.py                     # 命令行版（函数化拆分为 4 个阶段）
├── knowledge_base/                   # 知识库：放 .txt 就会自动被读入
│   ├── 产品手册.txt                   #   智净X3 完整产品手册（约 3100 字）
│   ├── 公司介绍.txt                   #   品牌背景 / 发展历程 / 核心优势
│   ├── 常见问题.txt                   #   20 条高频 FAQ
│   └── 智净X3产品推销话术.txt           #   导购 / 直播 / 朋友圈话术
├── product_manual.txt                # 历史遗留：V1 单文档版的知识库（保留作对比）
├── requirements.txt                  # 依赖
├── .devcontainer/                    # GitHub Codespaces 配置（自动装依赖 + 起 Streamlit）
├── .env                              # API Key（本地自建，已被 .gitignore 排除）
└── README.md
```

---

## ⚙️ 技术栈

| 环节 | 选型 | 说明 |
|---|---|---|
| 文本向量化 | `sentence-transformers` / `all-MiniLM-L6-v2` | 384 维，本地 CPU 运行，首次使用下载约 90MB |
| 相似度检索 | `numpy` 手写余弦相似度 | `np.dot(A, q) / (‖A‖·‖q‖)`，无向量数据库 |
| 大模型 | DeepSeek API（`deepseek-v4-flash`） | OpenAI 兼容的 ChatCompletions 接口 |
| Web 框架 | Streamlit | `@st.cache_resource` 缓存模型、`@st.cache_data` 缓存索引 |
| 其他 | `python-dotenv`、`requests` | 密钥管理与 HTTP 调用 |

---

## 🧠 它是怎么工作的（6 步流水线）

```
① 加载模型   SentenceTransformer('all-MiniLM-L6-v2')        ← 只加载一次（@st.cache_resource）
② 读库切块   glob('knowledge_base/*.txt') → 逐行扫描 → 切块 → 过滤过短块
③ 建立索引   model.encode(chunks) → 形状 (N, 384) 的矩阵      ← 只算一次（@st.cache_data）
④ 问题编码   encode("电池能用多久？") → 384 维向量
⑤ 相似度检索 余弦相似度排序 → 取 top-10 块 → 拼成 context
⑥ 约束生成   Prompt(人设 + 参考资料 + 问题) → DeepSeek → 回答
```

代码位置对照：`app.py` 中 模型加载（19-23 行）、切块（26-63 行）、向量化（69-73 行）、余弦检索（83-88 行）、top-k（91 行）、Prompt 与调用（95-126 行）、界面（129-142 行）。

`rag_folder.py` 里把四个阶段命名成了四个"岗位"，便于理解：

| 角色 | 函数 | 职责 |
|---|---|---|
| 仓库管理员 | `load_docs()` | 读文档、切块 |
| 指纹录入员 | `build_index()` | 把文字变成数字指纹（向量） |
| 接待员 | `retrieve()` | 听问题，去库里找最像的 top-k 段 |
| 秘书 | `generate()` | 拿着资料和问题去问大模型，写成答案 |

---

## 🔧 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/wanzitudou/python_practice.git
cd python_practice

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key：新建 .env 文件（该文件已被 .gitignore 排除，不会上传）
DEEPSEEK_API_KEY=sk-你的真实Key

# 4. 运行命令行版
python rag_folder.py

# 5. 运行网页版
streamlit run app.py
```

> 💡 国内网络：代码里已设置 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`，首次下载模型走镜像。

---

## 🧪 运行效果

<img width="780" height="535" alt="命令行版问答效果" src="https://github.com/user-attachments/assets/5581d958-77a5-4509-a3fc-cd558c58910c" />
<img width="756" height="253" alt="网页版问答效果" src="https://github.com/user-attachments/assets/78bae44f-98a8-41d4-90c1-964ca38ef685" />

**示例一（`rag_folder.py`）**

- 提问：`公司介绍`
- 检索命中：`公司背景：智净口腔科技是一家专注于声波电动牙刷及口腔健康管理系统的创新型科技企业，核心团队来自消费电子、电机控制和口腔医学领域……`
- AI 回答：根据参考资料概括公司定位、团队背景与核心优势

**示例二（`app.py`）**

- 提问：`充满电真的能用45天吗？`
- 检索命中：`标称续航：45天（每天使用2次，每次2分钟，标准日常清洁模式）……使用深层抛光模式 → 续航约 38~40天……`
- AI 回答：先给出 45 天的成立条件，再说明不同模式下的浮动范围

---

## 📝 版本演进

| 版本 | 文件 | 做法 | 为什么改 |
|---|---|---|---|
| **V1** | `app.py`（初版，62 行） | 把 `product_manual.txt` 全文塞进 Prompt | 先把链路跑通，但文档一长就爆上下文 |
| **V2** | `rag_folder.py` | 多文档 + 向量检索 + top-k 召回 | 只把最相关的几段交给模型，省 token、更准 |
| **V3** | `app.py`（现版，142 行） | 网页版升级为多文档，加入双层缓存与相对路径 | 让网页版和命令行版能力对齐，并解决"换机器跑不了""每次交互都重新加载模型" |

> V1 的 `product_manual.txt` 保留在仓库里，作为"从单文档到多文档"的对照。

---

## ⚠️ 已知限制（也是下一步的优化方向）

1. **向量模型对中文区分度不足**：`all-MiniLM-L6-v2` 是英文模型，中文文本的相似度分数会挤在 0.45~0.75 的窄区间，甚至出现"无关问题分数高于相关问题"的情况。→ 计划换成 `BAAI/bge-small-zh-v1.5` 或 `shibing624/text2vec-base-chinese`。
2. **没有相似度阈值**：top-1 分数很低时依然会送去生成。→ 计划加阈值，低于阈值直接返回"资料中未找到"，省下 API 调用。
3. **切块粒度偏粗**：按行/编号切块，最长块接近 1500 字，一个块里可能混了多个主题。→ 计划改为 250~500 字 + 50~80 字重叠，并保留表格行结构。
4. **回答不可溯源**：上下文里没有标注"来自哪个文件"，无法展示引用来源。→ 计划在回答中附带文件名与相似度。
5. **没有量化评估**：目前靠人工观察检索结果。→ 计划构造 20 条问答评测集，计算 hit@3 / MRR。

---

## 🧠 学习心得

这个项目是我学习 RAG 过程中的实践产出。从最初把整份手册塞进 Prompt，到实现多文档的向量检索与受约束生成，最大的体会是：**RAG 的瓶颈往往不在大模型，而在切块和检索**——切块切碎了，再强的模型也答不对。

在把整条链路手写一遍之后，我又用 Chroma 做了一遍向量持久化、用 LangChain + LangGraph 重做了一遍（RAG 作为 Agent 的一个工具），从而清楚地区分出"框架帮我做了什么"和"我真正需要理解什么"。

---

## 👤 作者

Meatball Potatoes
