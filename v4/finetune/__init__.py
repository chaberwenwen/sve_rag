"""
v4 嵌入模型微调模块

实现方案: HuggingFace Transformers + PEFT (LoRA) + 手写 MNRL 训练循环
面向影之诗 EVOLVE 卡牌规则领域的嵌入模型微调。

核心思路: 训练「效果文本 → QA问题」正例对，建立效果-问题语义桥梁。
当用户输入新卡效果时，通过效果相似度检索到已有 QA 的老卡牌相关问题。

技术要点:
  - 不依赖 sentence_transformers（避免 Windows DLL 兼容性问题）
  - 使用 transformers.AutoModel + AutoTokenizer 直接加载 BGE
  - mean_pooling (attention-weighted) + L2 normalize 与 BGE 官方一致
  - PEFT LoRA: r=16, alpha=32, target=['query','key','value','dense']
  - MNRL: batch内自动负例, scaled cosine (scale=20.0) → CE loss

模块组成:
  - prepare_data.py : 构造训练样本（--effect-only 仅生成 effect_to_question）
  - train_embedder.py : LoRA 微调主脚本（手动 MNRL 对比学习）
  - eval_embedder.py : 评估脚本（Recall@K / MRR，支持 A/B 对比）

使用方法:
  # 1. 生成训练数据
  python -m v4.finetune.prepare_data --effect-only

  # 2. 训练（dry-run 先检查环境）
  python -m v4.finetune.train_embedder --dry-run
  python -m v4.finetune.train_embedder

  # 3. 评估效果
  python -m v4.finetune.eval_embedder --compare   # A/B 对比

LoRA 开关 (config.py):
  LORA_ENABLED = False   # 原始 BGE，默认
  LORA_ENABLED = True    # BGE + LoRA adapter
"""
