"""
嵌入模型 LoRA 微调训练脚本 —— 效果文本 → QA问题

训练方法:
  - 直接使用 HuggingFace Transformers + PEFT（不依赖 SentenceTransformers）
  - LoRA 低秩适配 (PEFT)，仅训练 ~0.3% 参数
  - batch 内自动负例 MNRL：mean pooling → L2 normalize → scaled cosine → CE loss
  - 内置 train/val 划分 + 早停，防止过拟合

防过拟合策略:
  - 80/20 train/val 随机划分
  - 基于 val_loss 的早停（patience=5）
  - LoRA rank (r=16, alpha=32, dropout=0.1)
  - weight_decay=0.01 正则化
  - 每 epoch 打印 train_loss vs val_loss 对比
  - 仅保存 val_loss 最优的 checkpoint

使用方法:
  python -m v4.finetune.train_embedder             # 默认超参数训练
  python -m v4.finetune.train_embedder --epochs 20  # 自定义轮数
  python -m v4.finetune.train_embedder --dry-run    # 检查环境不训练
"""

import json
import os
import sys

from dataclasses import dataclass, field

# 路径设置
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V4_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_V4_DIR)
sys.path.insert(0, _V4_DIR)
sys.path.insert(0, _PROJECT_DIR)

from config import EMBED_MODEL_NAME

# ============================================================
# 输出路径
# ============================================================
FINETUNE_DATA_DIR = os.path.join(_V4_DIR, 'finetune', 'data')
MODEL_OUTPUT_DIR = os.path.join(_V4_DIR, 'finetune', 'model')

TRAIN_PAIRS_PATH = os.path.join(FINETUNE_DATA_DIR, 'train_pairs.json')


# ============================================================
# 训练配置
# ============================================================
@dataclass
class TrainConfig:
    """LoRA 微调超参数（经 A/B 评估调优）"""
    batch_size: int = 32
    learning_rate: float = 2e-5           # 恢复到原值（1e-5 欠拟合）
    num_epochs: int = 15                   # 上限轮数（早停会提前终止）
    warmup_steps: int = 100
    max_seq_length: int = 256
    lora_r: int = 16                       # 恢复到 r=16（r=8 欠拟合）
    lora_alpha: int = 32                   # 与 r=16 匹配
    lora_dropout: float = 0.1              # 适度 dropout（0.15 偏高）
    target_modules: tuple = ('query', 'key', 'value', 'dense')
    # 防过拟合（保留验证集 + 早停 + weight_decay）
    val_split: float = 0.20               # 验证集比例
    early_stopping_patience: int = 5       # val_loss 不降即停
    weight_decay: float = 0.01             # AdamW 权重衰减（新增）
    scale: float = 20.0                    # 恢复到原值（15 信号太弱）
    seed: int = 42                         # 随机种子（可复现划分）


def load_training_pairs(data_path: str = None) -> list[tuple[str, str]]:
    """
    加载训练数据，提取以下类型的正例对：

    effect_to_question:       主卡效果 → QA问题
    effect_to_question_swap:  被提及卡效果 → QA问题（多卡交互视角交换）
    qa_multi_card:            被提及卡 ←→ QA 正例对（question→answer）
    """
    path = data_path or TRAIN_PAIRS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'训练数据不存在: {path}\n'
            f'请先运行: python -m v4.finetune.prepare_data --effect-only --swap-multi'
        )

    with open(path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    accepted_types = {'effect_to_question', 'effect_to_question_swap'}
    pair_data = [
        (p['query'], p['document'])
        for p in pairs
        if p.get('type') in accepted_types
    ]

    type_counts = {}
    for p in pairs:
        t = p.get('type', '')
        if t in accepted_types:
            type_counts[t] = type_counts.get(t, 0) + 1
    print(f'加载训练数据: {len(pair_data)} 对 ('
          + ', '.join(f'{t}:{c}' for t, c in sorted(type_counts.items())) + ')')
    if not pair_data:
        raise ValueError('没有找到 effect_to_question / effect_to_question_swap / qa_multi_card 类型的训练对')
    return pair_data


# ============================================================
# Mean Pooling（与 BGE 官方一致：attention-weighted mean）
# ============================================================
def mean_pooling(token_embeddings, attention_mask):
    """
    attention-weighted mean pooling over all tokens.

    Args:
        token_embeddings: (B, L, D)
        attention_mask:  (B, L)
    Returns:
        (B, D)
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ============================================================
# 环境变量强制离线（在 import transformers 之前设置）
# ============================================================
def _set_offline_env():
    """设置 HuggingFace 离线环境变量，避免联网检查导致长时间卡住。"""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')


def _get_model_path():
    """确定模型加载路径：优先本地 v4/models/，其次 HF 缓存。"""
    local_path = os.path.join(_V4_DIR, 'models', 'bge-small-zh-v1.5')
    if os.path.isdir(local_path):
        return local_path
    return EMBED_MODEL_NAME


# ============================================================
# 训练主函数
# ============================================================
def _compute_batch_loss(batch: list[tuple[str, str]], model, tokenizer, device, cfg: TrainConfig):
    """
    计算一个 batch 的 MNRL loss（不更新梯度，用于验证集评估）。

    与训练前向完全一致的 pooling → normalize → scaled cosine → CE。
    """
    import torch

    queries = [p[0] for p in batch]
    docs = [p[1] for p in batch]

    q_enc = tokenizer(
        queries, padding=True, truncation=True,
        max_length=cfg.max_seq_length, return_tensors='pt',
    )
    d_enc = tokenizer(
        docs, padding=True, truncation=True,
        max_length=cfg.max_seq_length, return_tensors='pt',
    )
    q_enc = {k: v.to(device) for k, v in q_enc.items()}
    d_enc = {k: v.to(device) for k, v in d_enc.items()}

    with torch.no_grad():
        q_out = model(**q_enc)
        d_out = model(**d_enc)

        q_pooled = mean_pooling(q_out.last_hidden_state, q_enc['attention_mask'])
        d_pooled = mean_pooling(d_out.last_hidden_state, d_enc['attention_mask'])

        q_norm = torch.nn.functional.normalize(q_pooled, p=2, dim=1)
        d_norm = torch.nn.functional.normalize(d_pooled, p=2, dim=1)

        scores = torch.matmul(q_norm, d_norm.T) * cfg.scale
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, labels)

    return loss.item()


def _evaluate_val(val_data: list[tuple[str, str]], model, tokenizer, device, cfg: TrainConfig) -> float:
    """在整个验证集上计算平均 loss。"""
    import torch

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(val_data), cfg.batch_size):
        batch = val_data[i:i + cfg.batch_size]
        total_loss += _compute_batch_loss(batch, model, tokenizer, device, cfg)
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(config: TrainConfig = None):
    """
    执行 LoRA 微调训练（含 train/val 划分 + 早停防过拟合）。

    完全绕过 SentenceTransformers，使用 transformers.AutoModel + PEFT，
    手动实现 MNRL 训练循环。
    """
    cfg = config or TrainConfig()
    import random

    _check_environment()

    # ---- 限制线程（避免 Windows CPU 上的 OpenMP 冲突） ----
    import torch
    torch.set_num_threads(2)
    print(f'  torch threads: {torch.get_num_threads()}')

    # ---- 加载数据 ----
    pair_data = load_training_pairs()

    # ---- Train / Val 划分 ----
    import random as _random
    _random.seed(cfg.seed)
    indices = list(range(len(pair_data)))
    _random.shuffle(indices)
    val_size = max(1, int(len(pair_data) * cfg.val_split))
    train_indices = set(indices[val_size:])
    val_indices = set(indices[:val_size])

    train_data = [pair_data[i] for i in range(len(pair_data)) if i in train_indices]
    val_data = [pair_data[i] for i in range(len(pair_data)) if i in val_indices]
    print(f'  train/val 划分: {len(train_data)} / {len(val_data)} '
          f'({cfg.val_split:.0%} val, seed={cfg.seed})')

    # ---- 强制离线模式 ----
    _set_offline_env()
    model_path = _get_model_path()

    # ---- 加载 Tokenizer ----
    print(f'\n加载 Tokenizer: {model_path}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f'  vocab_size: {tokenizer.vocab_size}')

    # ---- 加载基础 Transformer 模型 ----
    print(f'\n加载基础模型: {model_path}')
    from transformers import AutoModel
    try:
        base_model = AutoModel.from_pretrained(model_path, local_files_only=True)
    except Exception:
        print('  本地加载失败，启用网络回退...')
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        base_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    print(f'  device: {device}')
    print(f'  hidden_size: {base_model.config.hidden_size}')

    # ---- 统计参数 ----
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f'  总参数: {total_params:,}')

    # ---- 注入 LoRA ----
    base_model = _inject_lora(base_model, cfg)

    trainable_params = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )
    print(
        f'可训练参数: {trainable_params:,} '
        f'({trainable_params / total_params * 100:.2f}%)'
    )

    # ---- 优化器 & 调度器 ----
    optimizer = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.num_epochs * ((len(train_data) + cfg.batch_size - 1) // cfg.batch_size)
    warmup_steps = min(cfg.warmup_steps, total_steps // 6)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 训练循环 ----
    print(f'\n开始训练（手动循环 + 验证集监控）...')
    print(f'  epochs: {cfg.num_epochs}')
    print(f'  batch_size: {cfg.batch_size}')
    print(f'  lr: {cfg.learning_rate}')
    print(f'  weight_decay: {cfg.weight_decay}')
    print(f'  warmup_steps: {warmup_steps}')
    print(f'  total_steps: {total_steps}')
    print(f'  scale: {cfg.scale}')
    print(f'  lora_r: {cfg.lora_r}, lora_alpha: {cfg.lora_alpha}, lora_dropout: {cfg.lora_dropout}')
    print(f'  early_stopping_patience: {cfg.early_stopping_patience}')
    print('-' * 50)

    global_step = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    base_model.train()

    for epoch in range(cfg.num_epochs):
        random.shuffle(train_data)

        train_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), cfg.batch_size):
            batch = train_data[i:i + cfg.batch_size]
            queries = [p[0] for p in batch]
            docs = [p[1] for p in batch]

            # ---- Tokenize ----
            q_enc = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=cfg.max_seq_length,
                return_tensors='pt',
            )
            d_enc = tokenizer(
                docs,
                padding=True,
                truncation=True,
                max_length=cfg.max_seq_length,
                return_tensors='pt',
            )

            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            d_enc = {k: v.to(device) for k, v in d_enc.items()}

            # ---- Forward: Transformer (PEFT-wrapped) ----
            q_out = base_model(**q_enc)
            d_out = base_model(**d_enc)

            # ---- Mean Pooling ----
            q_pooled = mean_pooling(q_out.last_hidden_state, q_enc['attention_mask'])
            d_pooled = mean_pooling(d_out.last_hidden_state, d_enc['attention_mask'])

            # ---- MNRL Loss: L2 normalize → scaled cosine → CE ----
            q_norm = torch.nn.functional.normalize(q_pooled, p=2, dim=1)
            d_norm = torch.nn.functional.normalize(d_pooled, p=2, dim=1)

            scores = torch.matmul(q_norm, d_norm.T) * cfg.scale
            labels = torch.arange(scores.size(0), device=scores.device)
            loss = torch.nn.functional.cross_entropy(scores, labels)

            # ---- Backward ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in base_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- 验证集评估 ----
        val_loss = _evaluate_val(val_data, base_model, tokenizer, device, cfg)

        lr_now = scheduler.get_last_lr()[0]
        overfit_gap = val_loss - avg_train_loss
        flag = ' *' if overfit_gap > 0.5 else ''
        best_mark = ' [BEST]' if val_loss < best_val_loss else ''

        print(f'Epoch {epoch + 1:2d}/{cfg.num_epochs} | '
              f'train_loss: {avg_train_loss:.4f} | val_loss: {val_loss:.4f} | '
              f'gap: {overfit_gap:+.4f}{flag} | lr: {lr_now:.2e}{best_mark}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            _save_adapter(base_model, MODEL_OUTPUT_DIR, is_best=True)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f'\n早停触发！val_loss 连续 {patience_counter} 轮未改善')
                break

    # ---- 最终保存（如果早停前没有任何最佳保存） ----
    if best_val_loss == float('inf'):
        _save_adapter(base_model, MODEL_OUTPUT_DIR, is_best=False)
        best_epoch = cfg.num_epochs

    print(f'\n训练完成！')
    print(f'  最佳 val_loss: {best_val_loss:.4f} (Epoch {best_epoch})')
    print(f'  早停 patience: {cfg.early_stopping_patience}')
    print(f'  LoRA adapter → {MODEL_OUTPUT_DIR}')
    print(f'  使用时设置 config.py 中 LORA_ENABLED = True 即可加载')


def _save_adapter(model, output_dir: str, is_best: bool = False):
    """保存 PEFT adapter 权重。"""
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
        tag = '最佳' if is_best else '最终'
        print(f'  [{tag}] adapter 已保存 → {output_dir}')
    else:
        print('  [WARNING] 模型没有 save_pretrained，尝试保存 state_dict...')
        import torch
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))


def _check_environment():
    """检查必需的依赖是否安装。"""
    missing = []

    try:
        import transformers
        print(f'transformers: {transformers.__version__}')
    except ImportError:
        missing.append('transformers')

    try:
        import peft
        print(f'peft: {peft.__version__}')
    except ImportError:
        missing.append('peft')

    try:
        import torch
        print(f'torch: {torch.__version__}')
    except ImportError:
        missing.append('torch')

    if missing:
        raise ImportError(
            f'缺少依赖: {", ".join(missing)}\n'
            f'请运行: pip install transformers peft torch'
        )


def _inject_lora(base_model, cfg: TrainConfig):
    """
    在 HuggingFace AutoModel 上注入 LoRA adapter。

    Args:
        base_model: HuggingFace AutoModel (BertModel)
        cfg: 训练配置

    Returns:
        PeftModel (LoRA-wrapped model)
    """
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(base_model, lora_config)

    print(f'  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, '
          f'target={cfg.target_modules}')

    return peft_model


def dry_run():
    """干运行：只检查环境和数据，不实际训练。"""
    print('=== Dry Run ===')
    _check_environment()
    pair_data = load_training_pairs()
    print(f'数据正常: {len(pair_data)} 对')

    if pair_data:
        q, d = pair_data[0]
        print(f'\n示例正例对:')
        print(f'  query ({len(q)} chars): {q[:120]}...')
        print(f'  doc   ({len(d)} chars): {d[:120]}...')

    # 设置离线模式，避免 dry-run 时也卡住
    _set_offline_env()
    model_path = _get_model_path()

    print(f'\n测试 Tokenizer 加载: {model_path}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    enc = tokenizer(['测试文本'], padding=True, truncation=True,
                    max_length=256, return_tensors='pt')
    print(f'  Tokenizer 正常, input_ids shape: {enc["input_ids"].shape}')

    print('\n环境检查通过，可以开始训练。')


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='LoRA 微调 Embedding 模型（效果文本→QA问题）'
    )
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数上限 (默认: 15，早停会提前终止)')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小 (默认: 32)')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率 (默认: 2e-5)')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank (默认: 16)')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='LoRA dropout (默认: 0.1)')
    parser.add_argument('--val-split', type=float, default=0.20, help='验证集比例 (默认: 0.20)')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心 (默认: 5)')
    parser.add_argument('--scale', type=float, default=20.0, help='MNRL 温度系数 (默认: 20)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减 (默认: 0.01)')
    parser.add_argument('--dry-run', action='store_true', help='只检查环境，不训练')
    parser.add_argument('--data', type=str, default=None, help='训练数据路径')
    parser.add_argument('--no-val', action='store_true', help='不使用验证集（全部数据用于训练，仅调试用）')
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        val_split=0.0 if args.no_val else args.val_split,
        early_stopping_patience=args.patience,
        weight_decay=args.weight_decay,
        scale=args.scale,
    )
    train(config)


if __name__ == '__main__':
    main()
