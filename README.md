# SVE RAG — 影之诗 EVOLVE 中文规则查询系统

基于 **RAG (Retrieval-Augmented Generation)** 架构的卡牌游戏规则问答系统。

## 架构

```
用户输入 → ① 名称索引识别卡牌 → ② 内容索引语义检索 → ③ 关联 QA → ④ 上下文构建 → ⑤ LLM 生成回答
```

- **双库 FAISS 索引**：名称索引（卡牌识别）+ 内容索引（语义检索）
- **Embedding**：BAAI/bge-small-zh-v1.5（512 维，中英文优化），支持 LoRA 微调
- **LLM**：DeepSeek API / Ollama 本地（qwen2.5:7b），后端可插拔
- **Web UI**：Gradio 界面，多 Tab 展示 RAG 中间结果

## 快速开始

```bash
# 安装依赖
cd v4 && pip install -r requirements.txt

# 构建索引
python v4/run.py index

# Web UI（推荐）
python v4/run.py ui

# 命令行查询
python v4/run.py cli "BP01-001 的效果是什么"
```

## 项目结构

```
sve_rag/
├── data/          # 卡牌 JSON、QA 翻译、规则文本
├── llm/           # LLM 客户端（多后端抽象）
├── ui/            # Gradio Web 界面
├── v4/            # RAG 核心（双库架构 + LoRA）
│   ├── rag/       # Embedder / Retriever / CardMatcher / Pipeline
│   ├── indexer/   # 索引构建
│   ├── finetune/  # LoRA 微调流水线
│   ├── models/    # 离线 Embedding 模型
│   └── index/     # FAISS 索引文件
└── scripts/       # 数据获取脚本（仅本地使用）
```

## 关键技术

- **FAISS IndexFlatIP** + L2 归一化 = 余弦相似度
- **BGE attention-weighted mean pooling**（非 [CLS]）
- **LoRA PEFT 注入**，运行时可切换基座/微调模型
- **MultipleNegativesRankingLoss** + BM25 困难负样本挖掘
- **双向量检索**：效果向量 + 问题向量合并去重
