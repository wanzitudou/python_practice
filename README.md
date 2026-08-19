# 📚 多文档智能问答助手

> 一个基于 RAG（检索增强生成）技术的本地知识库问答系统，能够从多个文档中检索信息并生成回答。

## 🚀 主要功能
- 支持从指定文件夹批量读取 `.txt` 文档
- 自动将长文档切分为语义块
- 使用 `sentence-transformers` 对文本块和用户问题进行向量化
- 通过余弦相似度检索与问题最相关的 3-5 个文本块
- 调用 DeepSeek API 生成高质量、基于资料的回答

## ⚙️ 技术栈
- `sentence-transformers`（本地向量化模型 `all-MiniLM-L6-v2`）
- `numpy` + `requests`
- `python-dotenv`（环境变量管理）
- DeepSeek API（大语言模型）

## 📁 项目结构
```
python训练/
├── rag_folder.py          # 主程序
├── knowledge_base/         # 存放 .txt 知识库
│   ├── 产品手册.txt
│   ├── 常见问题.txt
│   └── 公司介绍.txt
├── .env                    # 存放 DeepSeek API Key
└── requirements.txt        # 项目依赖
```

## 🧪 运行效果示例

**用户提问**：“公司介绍”

**系统检索到的相关段落**：
- “公司背景： 智净口腔科技是一家专注于声波电动牙刷...”
- “联系方式： 官方网站：www.zhi-jing.com...”
- “本公司产品“智净X3”电动牙刷的核心参数如下...”

**AI 回答**：
> 根据参考资料，智净口腔科技是一家专注于声波电动牙刷及口腔健康管理系统的创新型科技企业。核心团队来自消费电子、电机控制和口腔医学领域，拥有超过10年的行业经验...

## 🔧 快速开始
```bash
# 1. 安装依赖
pip install sentence-transformers numpy requests python-dotenv

# 2. 配置环境变量（在 .env 中填入你的 DeepSeek API Key）
DEEPSEEK_API_KEY=sk-你的Key

# 3. 运行脚本
python rag_folder.py
```

## 💡 适用场景
- 企业内部知识库问答
- 产品说明书智能检索
- 个人文档助手

## 📝 作者
Meatball Potatoes

