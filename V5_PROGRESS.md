# RAG V5 实验记录（进行中）

> 2026-11-07 起。**默认链路仍是 V4**；下面是对比实验，不是已上线版本。
> 原则：只写跑出来的数，不把“计划”写成“已完成”。

## 一、先扩评测集：15 条 → 50 条

V4 的 15 条正样本 + 4 条负样本太小，扩成：

- **40 条正样本**，覆盖 4 份知识库：产品手册 14、常见问题 12、公司介绍 7、推销话术 7；
- **10 条负样本**，包括知识库里没有的 CEO、颜色、电池品牌、App、双十一、股票代码等；
- 指标：hit@1 / hit@3 / hit@5、MRR@5、分文档 hit@5、正样本误拒、拒答 precision/recall。

脚本：`eval_retrieval_v5.py`；结果：`eval_results/baseline_v4_50.json`。

### V4 纯向量 baseline（200/40/90，bge-small-zh）

| 指标 | 数值 |
|---|---|
| hit@1 | **82.5%** |
| hit@3 | **95.0%** |
| hit@5 | **100%** |
| MRR@5 | **0.890** |
| 负样本拒答（阈值 0.40） | **3/10** |
| 正样本误拒 | 0/40 |

**关键发现**：扩了负样本之后，0.40 阈值只拦下 3/10。负样本 top1 最高 **0.647**
（“CEO 叫什么名字”），而正样本 top1 最低 **0.438**——**单一分数阈值无法分开**。
V4 的 4 条负样本太容易，掩盖了这个问题。

## 二、BM25 + RRF：这个语料上不划算

实现：`hybrid_retrieval.py`（jieba 分词 + rank_bm25 + RRF，向量与 BM25 权重可调）。
脚本：`eval_retrieval_v5.py --retriever hybrid --bm25-weight ...`。

| 方案 | hit@1 | hit@3 | hit@5 | MRR@5 | 负样本拒答 |
|---|---|---|---|---|---|
| 纯向量（baseline） | **82.5%** | **95.0%** | **100%** | **0.890** | 3/10 |
| hybrid，bm25 权重 0.3 | 80.0% | 87.5% | 97.5% | 0.859 | 4/10 |
| hybrid，bm25 权重 0.2 | 80.0% | 87.5% | 97.5% | 0.859 | 4/10 |
| hybrid，bm25 权重 0.1 | 80.0% | 87.5% | 100% | 0.865 | 3/10 |

**结论**：在这个 78 片段、自然语言问题为主的小语料上，BM25 把“电池”“刷头”这类高频词
当成强信号，反而把真正相关的片段挤出 top-5。**不默认开启**，保留开关；
等语料变大、专有名词/型号变多时再验证，不为了“架构完整”硬上混合检索。

## 三、LLM rerank + 可答性判断：指标最好，但贵

实现：`llm_rerank.py`（DeepSeek 对向量 top-8 重排，并输出 `answerable`）。
脚本：`eval_llm_rerank.py`；结果：`eval_results/llm_rerank_v5_50.json`。

| 指标 | V4 纯向量 | + LLM rerank |
|---|---|---|
| hit@1 | 82.5% | **100%** |
| hit@3 | 95.0% | **100%** |
| hit@5 | 100% | **100%** |
| MRR@5 | 0.890 | **1.000** |
| 负样本拒答 | 3/10 | **10/10** |
| 正样本误拒 | 0/40 | **0/40** |
| 拒答 precision / recall | 1.00 / 0.30 | **1.00 / 1.00** |
| 成本 | 0 | **47,460 token / 50 次查询**（约 950 token/次） |

**必须说清楚的边界**：
- 40 条正样本、10 条负样本仍然小，100% 不能当成“准确率”对外说；
- 排序和“是否可答”由同一个 LLM 判断，可能偏乐观；需要人工抽检校准；
- 延迟没单独测（体感每次多 1–2 秒），**没有做并发/压测**；
- 成本比纯向量高一个量级，所以只做**可选实验**，默认关闭；生产更合适的是
  cross-encoder（bge-reranker），本次因模型下载卡住（0 字节）没跑成。

## 四、当前决策

1. **默认仍走 V4 纯向量**，保证线上行为和 README 里的旧数据一致；
2. `hybrid` 和 `llm_rerank` 都是**开关**，不默认开；
3. 下一步优先级：
   - 补 cross-encoder rerank（网络恢复后下载 bge-reranker-base）；
   - 把 LLM 可答性判断只放在“灰区”（比如 top1 分数 0.40–0.65）以降低成本；
   - 查询改写 / Agentic 检索（最多 2 轮），用 50 条评测对比；
   - 索引落盘与增量更新；
   - 补延迟、token 成本的正式测量。

## 五、复现命令

```powershell
# V4 纯向量，50 条评测
python eval_retrieval_v5.py --verbose --json-out eval_results\baseline_v4_50.json

# BM25 + RRF
python eval_retrieval_v5.py --retriever hybrid --bm25-weight 0.3 `
  --json-out eval_results\hybrid_v5_50.json

# LLM rerank（需要 DEEPSEEK_API_KEY；结果会缓存）
python eval_llm_rerank.py --verbose --json-out eval_results\llm_rerank_v5_50.json
```

> `.env` 不提交；`eval_results/*cache*.json` 也不提交（里面是本地实验缓存）。
