# SVE RAG — 影之诗 EVOLVE 中文规则查询系统

基于 **RAG (Retrieval-Augmented Generation)** 架构的卡牌游戏规则问答系统。

## 项目概述

面向《影之诗 EVOLVE》卡牌游戏的领域知识问答系统，解决通用大模型缺乏卡牌领域知识、无法检索相关效果的 QA 和规则的问题。

- **卡牌识别**：用户用自然语言描述卡牌（如"那个入场曲抽牌的精灵从者"），系统通过 FAISS 向量索引精准定位到对应卡牌
- **规则检索**：将识别的卡牌效果文本 + 用户问题分别向量化，语义检索出最相关的官方规则 Q&A
- **知识增强生成**：检索结果作为上下文注入 LLM，确保回答有据可依、避免幻觉
- **Embedding 领域微调**：基于 LoRA 对 BGE 模型进行卡牌领域适配，提升检索命中率

## 架构

```
用户输入 → ① 名称索引识别卡牌 → ② 内容索引语义检索 → ③ 关联 QA → ④ 上下文构建 → ⑤ LLM 生成回答
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
