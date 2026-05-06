"""
训练数据准备 —— 从 QA/卡牌数据生成对比学习样本

面试要点：
  - 正例对: (QA问题, 对应答案) — 核心语义对齐信号
  - 正例对: (卡牌名+效果描述, 同卡牌名+效果) — 卡牌检索对齐
  - 正例对: (用户问题变体, QA答案) — LLM 改写增强泛化
  - BM25 困难负样本: 用 BM25 检索出与 query 部分相关但非正例的文档
  - 输出通用 JSON 格式 (query/document 对)，供 train_embedder / eval_embedder 使用

使用方法:
  python -m v4.finetune.prepare_data          → 仅正例对
  python -m v4.finetune.prepare_data --bm25   → 正例对 + BM25 困难负样本
  python -m v4.finetune.prepare_data --augment → 正例对 + LLM 改写增强
"""

import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Optional

# 将 v4/ 和项目根目录加入 path，使 data/loader 和 config 可导入
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V4_DIR = os.path.dirname(_SCRIPT_DIR)          # v4/
_PROJECT_DIR = os.path.dirname(_V4_DIR)         # 项目根目录
sys.path.insert(0, _V4_DIR)                     # v4/config.py 等
sys.path.insert(0, _PROJECT_DIR)                # data/loader.py 等

from data.loader import load_cards, load_qa_cn
from config import INDEX_DIR


# ============================================================
# 输出路径
# ============================================================
FINETUNE_DATA_DIR = os.path.join(_V4_DIR, 'finetune', 'data')
os.makedirs(FINETUNE_DATA_DIR, exist_ok=True)

TRAIN_PAIRS_PATH = os.path.join(FINETUNE_DATA_DIR, 'train_pairs.json')
HARD_NEG_PATH = os.path.join(FINETUNE_DATA_DIR, 'hard_negatives.pkl')
TRAIN_EXAMPLES_PATH = os.path.join(FINETUNE_DATA_DIR, 'train_examples.pkl')


# ============================================================
# 共享工具：多卡检测
# ============================================================

def _build_name_to_cardno(cards: list[dict]) -> dict[str, str]:
    """构建「卡牌中文名 → cardno」反向索引（同名取最早弹包）。"""
    name_to_cardno: dict[str, str] = {}
    for c in cards:
        name = c.get('name', '').strip()
        if name:
            existing = name_to_cardno.get(name)
            if existing is None or c['cardno'] < existing:
                name_to_cardno[name] = c['cardno']
    return name_to_cardno


def _find_mentioned_cards(
    question_text: str,
    main_name: str,
    name_to_cardno: dict[str, str],
) -> set[str]:
    """
    在问题文本中检测被提及的其他卡牌名称（按长度降序匹配，排除主卡自身）。

    返回被提及卡牌的 cardno 集合（不包含主卡自身）。
    """
    mentioned: set[str] = set()
    sorted_names = sorted(name_to_cardno.keys(), key=len, reverse=True)
    for name in sorted_names:
        if name == main_name:
            continue
        if name in question_text:
            mentioned.add(name_to_cardno[name])
    return mentioned


# ============================================================
# 正例对生成
# ============================================================

def build_qa_positive_pairs() -> list[dict]:
    """
    从 data/qa_cn/ 构建 QA 正例对。

    每对包含:
      - query: QA 问题文本
      - document: QA 答案文本
      - metadata: {cardno, card_name, source}

    Returns:
        [{query, document, type: 'qa', cardno, card_name}, ...]
    """
    qa_data = load_qa_cn()
    pairs = []

    for cardno, qas in qa_data.items():
        for qa in qas:
            question = qa.get('question_cn', '').strip()
            answer = qa.get('answer_cn', '').strip()
            card_name = qa.get('card_name', '')

            if not question or not answer:
                continue

            pairs.append({
                'query': question,
                'document': answer,
                'type': 'qa',
                'cardno': cardno,
                'card_name': card_name,
            })

    print(f'[QA正例对] 共 {len(pairs)} 对')
    return pairs


def build_effect_to_question_pairs() -> list[dict]:
    """
    效果文本 → QA问题 正例对。

    对每个有 QA 的卡牌：
      - query: 卡牌效果描述 (card.description)，纯效果，不含卡牌名
      - doc:   卡牌关联的 QA 问题 (question_cn)

    目的：建立「效果文本 → 问题描述」的语义桥梁。
    当用户输入新卡效果时，即使新卡没有 QA，也能通过效果相似度
    检索到已有 QA 的老卡牌相关问题。

    Returns:
        [{query, document, type: 'effect_to_question', cardno, card_name}, ...]
    """
    qa_data = load_qa_cn()
    cards = load_cards()
    card_map = {c['cardno']: c for c in cards}
    pairs = []

    for cardno, qas in qa_data.items():
        card = card_map.get(cardno)
        if not card:
            continue
        desc = card.get('description', '').strip()
        if not desc:
            continue

        for qa in qas:
            question = qa.get('question_cn', '').strip()
            if not question:
                continue
            pairs.append({
                'query': desc,
                'document': question,
                'type': 'effect_to_question',
                'cardno': cardno,
                'card_name': card.get('name', ''),
            })

    print(f'[效果→问题正例对] 共 {len(pairs)} 对')
    return pairs


def build_multi_card_swap_pairs() -> list[dict]:
    """
    多卡交互 QA 主卡视角交换增强 —— effect_to_question 视角。

    策略：
       对于含有多张卡牌名称的 QA 问题（如"用远古精灵选择弓箭手时..."），
       该 QA 原始关联主卡 = 弓箭手 (BP01-018)，但问题中也提到了远古精灵 (BP01-003)。

       增强做法：
         - 额外创建正例对：远古精灵的效果描述 → 同一个 QA 问题
         - 这样，当用户输入远古精灵的效果时，也能检索到这张"副卡"相关的 QA

    注意事项：
       - 只对确实出现在提问中的卡牌名称做交换（精确匹配卡牌中文名）
       - 被提及的卡牌必须有有效效果描述
       - 不生成主卡自身的重复对（effect_to_question 已涵盖）

    Returns:
        [{query, document, type: 'effect_to_question_swap', cardno, card_name,
          orig_cardno, orig_card_name}, ...]
    """
    qa_data = load_qa_cn()
    cards = load_cards()
    card_map = {c['cardno']: c for c in cards}
    name_to_cardno = _build_name_to_cardno(cards)

    pairs = []
    skipped_no_effect = 0

    for cardno, qas in qa_data.items():
        main_card = card_map.get(cardno)
        main_name = main_card.get('name', '') if main_card else ''

        for qa in qas:
            question = qa.get('question_cn', '').strip()
            if not question:
                continue

            mentioned_cardnos = _find_mentioned_cards(question, main_name, name_to_cardno)

            for other_cardno in mentioned_cardnos:
                other_card = card_map.get(other_cardno)
                if not other_card:
                    continue
                other_desc = other_card.get('description', '').strip()
                if not other_desc:
                    skipped_no_effect += 1
                    continue

                pairs.append({
                    'query': other_desc,
                    'document': question,
                    'type': 'effect_to_question_swap',
                    'cardno': other_cardno,
                    'card_name': other_card.get('name', ''),
                    'orig_cardno': cardno,
                    'orig_card_name': main_name,
                })

    print(
        f'[多卡视角交换(effect→question)] 共 {len(pairs)} 对'
        + (f' (跳过无效果:{skipped_no_effect})' if skipped_no_effect else '')
    )
    return pairs


def build_multi_card_qa_pairs() -> list[dict]:
    """
    多卡交互 QA 正例对 —— 为被提及的卡牌也创建 (question, answer) 正例对。

    策略：
       对于含有多张卡牌名称的 QA 问题，除了原始主卡拥有该 QA 正例对外，
       被问题文本中提及的其他卡牌也应与同一条 QA 形成正例对。

       例如：
         问题："用远古精灵选择弓箭手时，是否触发..."
         答案："是的，会触发。"
         原始：弓箭手(BP01-018) ←→ 这条 QA
         增强：远古精灵(BP01-003) ←→ 这条 QA

     目的：让模型学会「被提及的卡牌」也与这条 QA 相关，
           当用户查询被提及卡牌时，也能检索到这条交互 QA。

    Returns:
        [{query, document, type: 'qa_multi_card', cardno, card_name,
          orig_cardno, orig_card_name}, ...]
    """
    qa_data = load_qa_cn()
    cards = load_cards()
    card_map = {c['cardno']: c for c in cards}
    name_to_cardno = _build_name_to_cardno(cards)

    pairs = []
    skipped_no_effect = 0

    for cardno, qas in qa_data.items():
        main_card = card_map.get(cardno)
        main_name = main_card.get('name', '') if main_card else ''

        for qa in qas:
            question = qa.get('question_cn', '').strip()
            answer = qa.get('answer_cn', '').strip()
            if not question or not answer:
                continue

            mentioned_cardnos = _find_mentioned_cards(question, main_name, name_to_cardno)
            # 也检测答案中是否提到了其他卡
            mentioned_in_answer = _find_mentioned_cards(answer, main_name, name_to_cardno)
            mentioned_cardnos |= mentioned_in_answer

            for other_cardno in mentioned_cardnos:
                other_card = card_map.get(other_cardno)
                if not other_card:
                    continue
                # 不要求被提及卡必须有效果描述（QA 正例对不需要效果）
                pairs.append({
                    'query': question,
                    'document': answer,
                    'type': 'qa_multi_card',
                    'cardno': other_cardno,
                    'card_name': other_card.get('name', ''),
                    'orig_cardno': cardno,
                    'orig_card_name': main_name,
                })

    print(f'[多卡QA正例对] 共 {len(pairs)} 对')
    return pairs


def build_card_positive_pairs() -> list[dict]:
    """
    从卡牌数据构建卡牌正例对。

    策略:
      - 每个卡牌的「名称+效果」作为 query
      - 同一卡牌的「名称索引文本」作为 document
      - 这教会模型：卡牌名 + 效果描述 ≈ 同一张卡的不同表述

    同时，同一职业/类型的卡牌之间可以互为正例（弱正例），
    但这里先只做严格正例，避免引入噪声。
    """
    cards = load_cards()
    pairs = []

    for card in cards:
        cardno = card.get('cardno', '')
        name = card.get('name', '')
        desc = card.get('description', '')
        card_class = card.get('class', '')
        card_type = card.get('card_type', '')
        types_str = ', '.join(card.get('types', []))
        rarity = card.get('rarity', '')

        if not name:
            continue

        # 正例 1: 综合描述 ↔ 名称索引
        query_text = (
            f'卡牌: {name} ({cardno})\n'
            f'职业: {card_class}\n'
            f'类型: {card_type}\n'
            f'特征: {types_str}\n'
            f'稀有度: {rarity}\n'
            f'效果: {desc}'
        )
        doc_text = f'卡牌: {name} ({cardno})'

        pairs.append({
            'query': query_text,
            'document': doc_text,
            'type': 'card_self',
            'cardno': cardno,
            'card_name': name,
        })

        # 正例 2: 名称查询 → 效果描述（模拟用户搜卡牌名）
        pairs.append({
            'query': f'{name}的效果是什么',
            'document': query_text,
            'type': 'card_name_query',
            'cardno': cardno,
            'card_name': name,
        })

        # 正例 3: 名称查询变体
        pairs.append({
            'query': f'请解释{name}的能力',
            'document': query_text,
            'type': 'card_name_query',
            'cardno': cardno,
            'card_name': name,
        })

    print(f'[卡牌正例对] 共 {len(pairs)} 对')
    return pairs


def build_domain_positive_pairs() -> list[dict]:
    """
    构造领域感知的正例对，让模型更关注游戏机制关键词。

    为每张卡牌生成聚焦于费用的变化、效果破坏、伤害等关键机制的正例对。
    """
    cards = load_cards()
    pairs = []

    # 机制关键词模板
    mechanic_templates = {
        'cost': [
            '{name}的费用是多少',
            '{name}的消费变化',
            '使用{name}后费用如何变化',
        ],
        'destroy': [
            '{name}能破坏什么',
            '{name}的破坏效果',
            '{name}可以消灭哪些卡牌',
        ],
        'damage': [
            '{name}的伤害是多少',
            '{name}能造成多少伤害',
            '{name}的攻击力',
        ],
        'timing': [
            '{name}的效果什么时候触发',
            '{name}的入场曲效果',
            '{name}的谢幕曲效果',
        ],
    }

    for card in cards:
        name = card.get('name', '')
        desc = card.get('description', '')
        cardno = card.get('cardno', '')
        card_class = card.get('class', '')
        card_type = card.get('card_type', '')

        if not name or not desc:
            continue

        card_text = (
            f'卡牌: {name} ({cardno})\n'
            f'职业: {card_class}\n'
            f'类型: {card_type}\n'
            f'效果: {desc}'
        )

        # 根据描述内容选择相关的模板
        desc_lower = desc.lower()
        relevant_mechanics = []

        if any(kw in desc for kw in ['费用', '消费', 'cost', '费', 'PP']):
            relevant_mechanics.append('cost')
        if any(kw in desc for kw in ['破坏', '消灭', '破坏', 'destruction']):
            relevant_mechanics.append('destroy')
        if any(kw in desc for kw in ['伤害', 'damage', '攻击', '给予']):
            relevant_mechanics.append('damage')
        if any(kw in desc for kw in ['入场曲', '谢幕曲', '进化时', '启动', '时机', '回合']):
            relevant_mechanics.append('timing')

        # 如果没有匹配到任何机制，至少用 cost 模板
        if not relevant_mechanics:
            relevant_mechanics = ['cost']

        for mech in relevant_mechanics:
            for template in mechanic_templates[mech]:
                query = template.format(name=name)
                pairs.append({
                    'query': query,
                    'document': card_text,
                    'type': f'domain_{mech}',
                    'cardno': cardno,
                    'card_name': name,
                })

    print(f'[领域正例对] 共 {len(pairs)} 对')
    return pairs


# ============================================================
# 困难负样本挖掘 (BM25)
# ============================================================

def _build_document_corpus() -> tuple[list[str], list[dict]]:
    """
    构建文档语料库用于 BM25 检索。

    包含:
      - 所有 QA 答案文本
      - 所有卡牌效果描述
      - 所有规则块文本

    Returns:
        (texts, metadatas) 两个等长列表
    """
    texts = []
    metadatas = []

    # QA 答案
    qa_data = load_qa_cn()
    for cardno, qas in qa_data.items():
        for qa in qas:
            answer = qa.get('answer_cn', '').strip()
            if answer:
                texts.append(answer)
                metadatas.append({
                    'type': 'qa_answer',
                    'cardno': cardno,
                    'card_name': qa.get('card_name', ''),
                    'question': qa.get('question_cn', ''),
                })

    # 卡牌效果
    cards = load_cards()
    for card in cards:
        desc = card.get('description', '')
        if desc:
            texts.append(desc)
            metadatas.append({
                'type': 'card_effect',
                'cardno': card.get('cardno', ''),
                'card_name': card.get('name', ''),
            })

    # 规则文本（如果存在）
    from data.loader import load_rules
    rules = load_rules()
    for rule in rules:
        rule_text = rule.get('text', '').strip()
        if rule_text:
            texts.append(rule_text)
            metadatas.append({
                'type': 'rule',
                'chapter': rule.get('chapter', ''),
                'section': rule.get('section', ''),
            })

    print(f'[BM25语料库] 共 {len(texts)} 篇文档')
    return texts, metadatas


def mine_hard_negatives(
    queries: list[str],
    query_cardnos: list[str],
    top_k: int = 10,
) -> list[list[dict]]:
    """
    用 BM25 为每个 query 挖掘困难负样本。

    对每个 query:
      1. BM25 检索 top-K 文档
      2. 排除正例（相同 cardno 的 QA 答案）
      3. 剩余 top-N 作为困难负样本

    Args:
        queries: 查询文本列表
        query_cardnos: 每个 query 对应的 cardno（用于排除正例）
        top_k: BM25 检索数量

    Returns:
        [ [{text, metadata, bm25_score}, ...], ... ] 每个 query 的困难负样本列表
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print('[WARNING] rank_bm25 未安装，跳过困难负样本挖掘')
        print('  安装: pip install rank_bm25')
        return [[] for _ in queries]

    import jieba

    corpus_texts, corpus_metadatas = _build_document_corpus()

    # 分词
    def tokenize(text: str) -> list[str]:
        return list(jieba.cut(text))

    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    all_hard_negs = []

    for query, cardno in zip(queries, query_cardnos):
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # 按分数降序排列
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        hard_negs = []
        for idx, score in ranked:
            meta = corpus_metadatas[idx]

            # 排除同 cardno 的正例
            if meta.get('cardno') == cardno:
                continue
            # 排除同一 QA 的问题（正例）
            if meta.get('type') == 'qa_answer' and meta.get('cardno') == cardno:
                continue

            hard_negs.append({
                'text': corpus_texts[idx],
                'metadata': meta,
                'bm25_score': float(score),
            })

            if len(hard_negs) >= 5:  # 每个 query 取 5 个困难负样本
                break

        all_hard_negs.append(hard_negs)

    total = sum(len(hn) for hn in all_hard_negs)
    print(f'[BM25困难负样本] 共挖掘 {total} 个 (平均 {total/len(queries):.1f}/query)')
    return all_hard_negs


# ============================================================
# 数据增强：LLM 改写 query
# ============================================================

def augment_queries(pairs: list[dict], n_variants: int = 2) -> list[dict]:
    """
    使用 LLM 对 query 做同义改写，增加训练数据多样性。

    由于这需要 LLM API 调用，这里只提供框架。
    实际使用时建议用 DeepSeek 批量改写。

    Args:
        pairs: 原始正例对列表
        n_variants: 每个 query 生成几个变体

    Returns:
        扩充后的正例对列表（含原始 + 变体）
    """
    # 这里仅做简单的模板替换，实际 LLM 调用可在运行时替换
    # 变体模板（中文同义表达）
    variant_templates = [
        '请帮我查一下，{q}',
        '我想了解：{q}',
        '{q}，请问这是什么意思',
        '关于{q}，规则是怎么说的',
        '{q}，能解释一下吗',
    ]

    augmented = list(pairs)
    for pair in pairs:
        query = pair['query']
        for i, tmpl in enumerate(variant_templates[:n_variants]):
            if '{q}' in tmpl:
                new_query = tmpl.format(q=query)
            else:
                new_query = query + tmpl

            new_pair = dict(pair)
            new_pair['query'] = new_query
            new_pair['type'] = pair['type'] + '_aug'
            augmented.append(new_pair)

    print(f'[数据增强] {len(pairs)} → {len(augmented)} 对 (+{len(augmented) - len(pairs)})')
    return augmented


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='准备微调训练数据（正例对 + BM25 困难负样本）'
    )
    parser.add_argument(
        '--bm25',
        action='store_true',
        help='启用 BM25 困难负样本挖掘（需要 pip install rank_bm25 jieba）',
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='启用 LLM 同义改写增强',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='输出 JSON 路径 (默认: v4/finetune/data/train_pairs.json)',
    )
    parser.add_argument(
        '--effect-only',
        action='store_true',
        help='仅生成 effect_to_question 正例对（效果文本→QA问题）',
    )
    args = parser.parse_args()

    print('=' * 60)
    print('微调训练数据准备')
    print('=' * 60)

    # Step 1: 构建正例对
    print('\n[Step 1] 构建正例对...')
    all_pairs = []

    if args.effect_only:
        # 仅效果→问题类型
        all_pairs.extend(build_effect_to_question_pairs())
    else:
        all_pairs.extend(build_qa_positive_pairs())
        all_pairs.extend(build_card_positive_pairs())
        all_pairs.extend(build_domain_positive_pairs())

    # 多卡视角交换：默认启用，为被提及的卡牌也生成正例对
    print('\n[多卡增强] 检测被引用卡牌并生成额外正例对...')
    all_pairs.extend(build_multi_card_swap_pairs())     # effect→question 视角交换
    all_pairs.extend(build_multi_card_qa_pairs())       # QA 正例对交换

    # Step 2: 数据增强（可选）
    if args.augment:
        print('\n[Step 2] 数据增强...')
        all_pairs = augment_queries(all_pairs, n_variants=2)

    # Step 3: 保存正例对
    print(f'\n[Step 3] 保存训练数据...')
    output_path = args.output or TRAIN_PAIRS_PATH
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f'  正例对已保存 → {output_path}')
    print(f'  总计 {len(all_pairs)} 对')

    # Step 4: BM25 困难负样本（可选）
    if args.bm25:
        print('\n[Step 4] BM25 困难负样本挖掘...')
        queries = [p['query'] for p in all_pairs]
        cardnos = [p.get('cardno', '') for p in all_pairs]
        hard_negs = mine_hard_negatives(queries, cardnos, top_k=10)

        with open(HARD_NEG_PATH, 'wb') as f:
            pickle.dump(hard_negs, f)
        print(f'  困难负样本已保存 → {HARD_NEG_PATH}')

    # Step 5: 统计信息
    print('\n' + '=' * 60)
    print('数据统计:')
    type_counts = {}
    for p in all_pairs:
        t = p['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f'  {t}: {c} 对')
    print(f'  合计: {len(all_pairs)} 对')
    print('=' * 60)


if __name__ == '__main__':
    main()
