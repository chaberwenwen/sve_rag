"""
嵌入模型评估脚本 —— 对比有无 LoRA 的检索效果

评估指标:
  - Recall@K (K=1,3,5,10): 正例 document 出现在 top-K 中的比例
  - MRR (Mean Reciprocal Rank): 正例排名的倒数均值

使用方法:
  python -m v4.finetune.eval_embedder              # 用 hold-out 评估
  python -m v4.finetune.eval_embedder --no-lora    # 只评估原始 BGE
  python -m v4.finetune.eval_embedder --lora       # 只评估 LoRA 版本
  python -m v4.finetune.eval_embedder --compare    # A/B 对比输出
"""

import json
import os
import sys
import time

from dataclasses import dataclass, field

# 路径设置
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V4_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_V4_DIR)
sys.path.insert(0, _V4_DIR)
sys.path.insert(0, _PROJECT_DIR)

import numpy as np

# ============================================================
# 路径
# ============================================================
FINETUNE_DATA_DIR = os.path.join(_V4_DIR, 'finetune', 'data')
TRAIN_PAIRS_PATH = os.path.join(FINETUNE_DATA_DIR, 'train_pairs.json')


# ============================================================
# 评估指标计算
# ============================================================
@dataclass
class EvalResult:
    """评估结果"""
    use_lora: bool
    num_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_rank: float
    encode_time_sec: float = 0.0


def compute_metrics(
    ranks: list[int],
    num_docs: int,
    encode_time: float = 0.0,
    use_lora: bool = False,
) -> EvalResult:
    """
    计算检索评估指标。

    Args:
        ranks: 每个 query 的正确答案排名 (1-based, -1 表示未找到)
        num_docs: 文档总数
        encode_time: 编码耗时
        use_lora: 是否使用 LoRA

    Returns:
        EvalResult 包含所有指标
    """
    valid_ranks = [r for r in ranks if r > 0]
    n = len(ranks)

    return EvalResult(
        use_lora=use_lora,
        num_queries=n,
        recall_at_1=sum(1 for r in ranks if 0 < r <= 1) / n if n else 0,
        recall_at_3=sum(1 for r in ranks if 0 < r <= 3) / n if n else 0,
        recall_at_5=sum(1 for r in ranks if 0 < r <= 5) / n if n else 0,
        recall_at_10=sum(1 for r in ranks if 0 < r <= 10) / n if n else 0,
        mrr=sum(1.0 / r for r in valid_ranks) / n if n else 0,
        mean_rank=sum(valid_ranks) / len(valid_ranks) if valid_ranks else float('inf'),
        encode_time_sec=encode_time,
    )


def evaluate_on_pairs(
    pairs: list[tuple[str, str]],
    use_lora: bool,
) -> EvalResult:
    """
    在给定的正例对上评估检索效果。

    流程:
      1. 收集所有唯一的 document，构建文档语料库
      2. 编码所有 document → document embeddings
      3. 对每个 query:
         a. 编码 query
         b. 计算与所有 document 的余弦相似度
         c. 记录正例 document 的排名

    注: 使用 rag.embedder 模块（不依赖 sentence_transformers），
        encode_texts() 内部完成 mean pooling + L2 normalize。
    """
    from rag.embedder import get_embedder, reload_embedder, encode_texts

    # 设置 LoRA 开关
    import config
    config.LORA_ENABLED = use_lora
    reload_embedder()

    # 收集所有唯一 document
    doc_set = {}
    doc_list = []
    for query, document in pairs:
        if document not in doc_set:
            doc_set[document] = len(doc_list)
            doc_list.append(document)

    print(f'文档语料: {len(doc_list)} 个唯一文档')

    # 编码所有 document（使用 embedder.encode_texts，与 RAG 检索一致）
    t0 = time.time()
    doc_embeddings = encode_texts(doc_list, normalize=True)
    encode_time = time.time() - t0
    print(f'文档编码耗时: {encode_time:.1f}s')

    # 逐 query 检索
    ranks = []
    batch_queries = [q for q, _ in pairs]

    print(f'评估 {len(batch_queries)} 个 query...')
    t1 = time.time()
    query_embeddings = encode_texts(batch_queries, normalize=True)
    query_encode_time = time.time() - t1
    encode_time += query_encode_time

    for i, (query, document) in enumerate(pairs):
        query_vec = query_embeddings[i]

        # 余弦相似度（向量已归一化，内积 = 余弦相似度）
        similarities = np.dot(doc_embeddings, query_vec)

        # 按相似度降序排名
        ranked_indices = np.argsort(-similarities)

        # 找正例 document 的排名
        target_idx = doc_set[document]
        for rank, idx in enumerate(ranked_indices):
            if idx == target_idx:
                ranks.append(rank + 1)  # 1-based
                break
        else:
            ranks.append(-1)  # 未找到（不应发生）

    return compute_metrics(ranks, len(doc_list), encode_time, use_lora)


def load_eval_pairs(data_path: str = None, hold_out: float = 0.0,
                    seed: int = 42) -> tuple:
    """
    加载评估用的正例对，支持按比例划分 train/val（用于检测过拟合）。

    Args:
        data_path: 数据文件路径
        hold_out: 验证集比例 (0.0 = 全部用作评估, 0.2 = 80/20 划分)
        seed: 随机种子（可复现划分）

    Returns:
        (train_pairs, val_pairs) 当 hold_out > 0 时返回两组，
        当 hold_out = 0 时 val_pairs 为空列表。
    """
    import random

    path = data_path or TRAIN_PAIRS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'评估数据不存在: {path}\n'
            f'请先运行: python -m v4.finetune.prepare_data --effect-only'
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
    print(f'加载评估数据: {len(pair_data)} 对 ('
          + ', '.join(f'{t}:{c}' for t, c in sorted(type_counts.items())) + ')')

    if hold_out <= 0:
        return pair_data, []

    # Train / Val 划分
    rng = random.Random(seed)
    indices = list(range(len(pair_data)))
    rng.shuffle(indices)
    val_size = max(1, int(len(pair_data) * hold_out))
    val_indices = set(indices[:val_size])
    train_pairs = [p for i, p in enumerate(pair_data) if i not in val_indices]
    val_pairs = [p for i, p in enumerate(pair_data) if i in val_indices]
    print(f'  划分 train/val: {len(train_pairs)} / {len(val_pairs)} '
          f'(hold_out={hold_out:.0%}, seed={seed})')
    return train_pairs, val_pairs


def format_result(result: EvalResult) -> str:
    """格式化评估结果为可读字符串。"""
    label = 'LoRA' if result.use_lora else '原始BGE'
    lines = [
        f'\n{"="*50}',
        f'  评估结果: {label}',
        f'{"="*50}',
        f'  查询数:     {result.num_queries}',
        f'  Recall@1:   {result.recall_at_1:.4f}',
        f'  Recall@3:   {result.recall_at_3:.4f}',
        f'  Recall@5:   {result.recall_at_5:.4f}',
        f'  Recall@10:  {result.recall_at_10:.4f}',
        f'  MRR:        {result.mrr:.4f}',
        f'  平均排名:   {result.mean_rank:.1f}',
        f'  编码耗时:   {result.encode_time_sec:.1f}s',
        f'{"="*50}',
    ]
    return '\n'.join(lines)


def compare_results(result_no_lora: EvalResult, result_lora: EvalResult) -> str:
    """生成 A/B 对比报告。"""
    def delta(new, old):
        if old == 0:
            return '+∞' if new > 0 else '0'
        return f'{((new - old) / old) * 100:+.1f}%'

    lines = [
        f'\n{"="*70}',
        f'  A/B 对比: 原始 BGE vs LoRA 微调',
        f'{"="*70}',
        f'  {"指标":<15} {"原始BGE":<12} {"LoRA":<12} {"变化":<10}',
        f'  {"-"*49}',
        f'  {"Recall@1":<15} {result_no_lora.recall_at_1:<12.4f} '
        f'{result_lora.recall_at_1:<12.4f} '
        f'{delta(result_lora.recall_at_1, result_no_lora.recall_at_1):>10}',
        f'  {"Recall@3":<15} {result_no_lora.recall_at_3:<12.4f} '
        f'{result_lora.recall_at_3:<12.4f} '
        f'{delta(result_lora.recall_at_3, result_no_lora.recall_at_3):>10}',
        f'  {"Recall@5":<15} {result_no_lora.recall_at_5:<12.4f} '
        f'{result_lora.recall_at_5:<12.4f} '
        f'{delta(result_lora.recall_at_5, result_no_lora.recall_at_5):>10}',
        f'  {"Recall@10":<15} {result_no_lora.recall_at_10:<12.4f} '
        f'{result_lora.recall_at_10:<12.4f} '
        f'{delta(result_lora.recall_at_10, result_no_lora.recall_at_10):>10}',
        f'  {"MRR":<15} {result_no_lora.mrr:<12.4f} '
        f'{result_lora.mrr:<12.4f} '
        f'{delta(result_lora.mrr, result_no_lora.mrr):>10}',
        f'  {"平均排名":<15} {result_no_lora.mean_rank:<12.1f} '
        f'{result_lora.mean_rank:<12.1f} '
        f'{delta(result_lora.mean_rank, result_no_lora.mean_rank):>10}',
        f'{"="*70}',
    ]
    return '\n'.join(lines)


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='评估 Embedding 模型检索效果（支持 A/B 对比 + 过拟合检测）'
    )
    parser.add_argument(
        '--no-lora', action='store_true',
        help='只评估原始 BGE 模型',
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='只评估 LoRA 微调版本',
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='A/B 对比原始 BGE 和 LoRA 版本',
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='评估数据路径',
    )
    parser.add_argument(
        '--hold-out', type=float, default=0.0,
        help='验证集比例，用于检测过拟合 (默认: 0.0 = 全部评估)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子 (默认: 42)',
    )
    args = parser.parse_args()

    # 默认行为：同时评估两者（含对比报告）
    run_no_lora = args.no_lora or not args.lora  # 除非只指定 --lora
    run_lora = args.lora or not args.no_lora      # 除非只指定 --no-lora
    run_compare = args.compare or (not args.no_lora and not args.lora)

    train_pairs, val_pairs = load_eval_pairs(args.data, args.hold_out, args.seed)

    if args.hold_out > 0:
        print(f'\n>>> 过拟合检测模式 (hold_out={args.hold_out})')
        print(f'    训练集 (用于评估): {len(train_pairs)} 对')
        print(f'    验证集 (未见过):   {len(val_pairs)} 对')
        all_datasets = [('训练集', train_pairs), ('验证集', val_pairs)]
    else:
        all_datasets = [('全量数据', train_pairs)]

    for ds_name, pairs in all_datasets:
        print(f'\n{"="*50}')
        print(f'  [{ds_name}] {len(pairs)} 对')
        print(f'{"="*50}')

        results = {}

        if run_no_lora:
            print(f'\n>>> [{ds_name}] 评估原始 BGE...')
            results['no_lora'] = evaluate_on_pairs(pairs, use_lora=False)
            print(format_result(results['no_lora']))

        if run_lora:
            print(f'\n>>> [{ds_name}] 评估 LoRA 版本...')
            results['lora'] = evaluate_on_pairs(pairs, use_lora=True)
            print(format_result(results['lora']))

        if run_compare and 'no_lora' in results and 'lora' in results:
            print(compare_results(results['no_lora'], results['lora']))


if __name__ == '__main__':
    main()
