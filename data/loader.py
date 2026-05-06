"""
数据加载层 —— 从 data/ 统一加载卡牌、QA、规则数据

面试要点：
  - 单一职责：只管「从哪里读 + 怎么映射」，不关心「怎么用」
  - 只加载纯数字编号的普通卡（如 BP01-001），不加载 SL/U/T 等特殊卡
  - 字段映射封装为独立函数，方便后续数据格式变更时只改一处
  - data/ 是共享模块，所有版本（v3/v4/...）共用同一套数据加载逻辑
"""

import os
import re
import json

# data/loader.py 位于 sve_rag/data/loader.py
# 需要找到 sve_rag/data/ 目录
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CARDS_DIR = os.path.join(_DATA_DIR, 'cards')
DATA_QA_CN_DIR = os.path.join(_DATA_DIR, 'raw', 'qa_en_translated')
DATA_RULES_DIR = os.path.join(_DATA_DIR, 'rules')


# ============================================================
# 辅助函数：判断是否为纯数字编号的普通卡
# ============================================================
def is_base_cardno(card_no: str) -> bool:
    """
    判断是否为「纯数字编号」的普通卡（如 BP01-002）。
    非普通卡包含字母后缀，如 BP01-SL02（异画）、BP01-U01（UR）、BP01-T01（Token）。
    """
    return bool(re.match(r'^\w{2}\d+-\d+$', card_no))


# ============================================================
# 字段映射：sve_rag_data 格式 → 内部统一格式
# ============================================================
def map_card_field(card: dict) -> dict:
    """
    将 sve_rag_data 的卡牌字段映射为内部统一格式。

    映射关系：
      card_no   → cardno        卡牌编号 (BP01-001)
      name_cn   → name          中文名称 (玫瑰皇后)
      desc_cn   → description   中文效果描述
      craft     → class         职业
      card_type → card_type     卡牌类型
      type      → types         种族/特征 (拆分为list)
      rare      → rarity        稀有度
      from      → pack          弹包
      cost      → cost          费用
      attack    → power         攻击力
      life      → hp            生命值
    """
    raw_type = card.get('type', '')
    types_list = [t.strip() for t in raw_type.split('/')] if raw_type else []

    return {
        'cardno': card.get('card_no', ''),
        'name': card.get('name_cn', ''),
        'class': card.get('craft', ''),
        'card_type': card.get('card_type', ''),
        'types': types_list,
        'rarity': card.get('rare', ''),
        'cost': card.get('cost', None),
        'power': card.get('attack', None),
        'hp': card.get('life', None),
        'description': card.get('desc_cn', ''),
        'pack': card.get('from', ''),
    }


# ============================================================
# 卡牌加载（纯数字普通卡，无需去重）
# ============================================================
def load_cards() -> list[dict]:
    """
    加载所有纯数字编号的普通卡牌数据。

    自动扫描 data/cards/ 下所有弹包目录，加载每个目录的 {EXP}_all_cards.json，
    只保留编号格式为 {EXP}-{DIGITS} 的普通卡（不含 SL/U/T 等特殊卡）。

    由于每张卡有唯一编号，无需稀有度去重。

    Returns:
        cards — 已做字段映射的卡牌列表
    """
    if not os.path.exists(DATA_CARDS_DIR):
        print(f'[WARNING] 卡牌数据目录不存在: {DATA_CARDS_DIR}')
        return []

    cards = []
    raw_total = 0
    filtered_total = 0

    for exp_dir in sorted(os.listdir(DATA_CARDS_DIR)):
        exp_path = os.path.join(DATA_CARDS_DIR, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        json_path = os.path.join(exp_path, f'{exp_dir}_all_cards.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_cards = json.load(f)

        for card in raw_cards:
            raw_total += 1
            card_no = card.get('card_no', '')
            # 只保留纯数字编号的普通卡
            if is_base_cardno(card_no):
                cards.append(map_card_field(card))
                filtered_total += 1

    print(f'加载 {len(cards)} 张卡牌（过滤特殊卡后，原始 {raw_total} 条目）')
    if raw_total > filtered_total:
        print(f'  已过滤 {raw_total - filtered_total} 条特殊卡（SL/U/T 等）')
    return cards


# ============================================================
# 中文 QA 加载
# ============================================================
def load_qa_cn() -> dict[str, list[dict]]:
    """
    加载中文 QA 数据（来自 data/raw/qa_en_translated/ 英文官方 QA 翻译版）。

    自动扫描 data/raw/qa_en_translated/ 下所有 BP*_qa.json 文件并合并。
    例如: BP01_qa.json, BP02_qa.json 等。

    重要：英文 QA 文件中的编号（如 BP11-001）与中文卡牌编号可能不一致。
    因此使用每条 QA 的 card_name 字段，通过卡牌中文名匹配到正确的中文 cardno。

    Returns:
        cardno → QA列表 的映射字典（key 为中文卡牌编号，合并所有弹数）
    """
    if not os.path.exists(DATA_QA_CN_DIR):
        print(f'[WARNING] 中文 QA 目录不存在: {DATA_QA_CN_DIR}')
        return {}

    # 构建「卡牌中文名 → 中文 cardno」的映射（同名取最小编号 = 基础卡）
    name_to_cardno = _build_name_to_cardno_mapping()

    merged = {}
    files_found = 0
    resolved_total = 0
    unresolved_total = 0
    unresolved_names = set()

    for fname in sorted(os.listdir(DATA_QA_CN_DIR)):
        if not fname.endswith('_qa.json'):
            continue
        qa_path = os.path.join(DATA_QA_CN_DIR, fname)
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f'[WARNING] 无法读取 QA 文件 {fname}: {e}')
            continue

        # 按 card_name 匹配到中文 cardno 再合并
        for en_cardno, qas in data.items():
            for qa in qas:
                card_name = qa.get('card_name', '').strip()
                if card_name:
                    cn_cardno = name_to_cardno.get(card_name)
                    if cn_cardno:
                        if cn_cardno not in merged:
                            merged[cn_cardno] = []
                        merged[cn_cardno].append(qa)
                        resolved_total += 1
                    else:
                        # 卡名无法匹配到中文 cardno，回退使用原英文编号
                        unresolved_total += 1
                        unresolved_names.add(card_name)
                        if en_cardno not in merged:
                            merged[en_cardno] = []
                        merged[en_cardno].append(qa)
                else:
                    # 没有 card_name 字段，回退使用原英文编号
                    if en_cardno not in merged:
                        merged[en_cardno] = []
                    merged[en_cardno].append(qa)

        files_found += 1
        qa_count = sum(len(v) for v in data.values())
        print(f'  加载 {fname}: {qa_count} 条 QA')

    if unresolved_total > 0:
        print(f'[WARNING] 共 {unresolved_total} 条 QA 无法通过卡名匹配中文编号，'
              f'已回退使用英文编号')
        print(f'  未能匹配的卡名: {sorted(unresolved_names)[:20]}')

    if files_found == 0:
        print('[WARNING] 未找到任何 BP*_qa.json 翻译文件，请先运行 scripts/translate_en_qa.py')
        return {}

    total = sum(len(v) for v in merged.values())
    print(f'合并后共 {total} 条中文 QA（其中 {resolved_total} 条通过卡名匹配），'
          f'涉及 {len(merged)} 张卡')
    return merged


def _build_name_to_cardno_mapping() -> dict[str, str]:
    """
    构建「卡牌中文名 → cardno」的映射，同名卡优先选基础形态（非进化卡）。

    策略：
      1. 优先选 cost >= 0 的卡（基础卡，排除进化形态 cost=-1）
      2. 同条件下选编号最小的
    """
    name_to_cardno: dict[str, str] = {}
    name_to_cost: dict[str, int] = {}
    cards = load_cards()
    for card in cards:
        name = card.get('name', '').strip()
        cardno = card.get('cardno', '')
        cost = card.get('cost', 0) or 0
        if not name or not cardno:
            continue
        existing = name_to_cardno.get(name)
        if existing is None:
            name_to_cardno[name] = cardno
            name_to_cost[name] = cost
        else:
            existing_cost = name_to_cost.get(name, 0)
            # 优先选基础形态（cost >= 0），其次选编号更小
            if ((cost >= 0 and existing_cost < 0)
                    or (cost >= 0 and existing_cost >= 0 and cardno < existing)
                    or (cost < 0 and existing_cost < 0 and cardno < existing)):
                name_to_cardno[name] = cardno
                name_to_cost[name] = cost
    return name_to_cardno


# ============================================================
# 规则文本加载
# ============================================================
def load_rules() -> list[dict]:
    """
    加载中文规则文本块（rules_chunks.zh.json）。

    每个 chunk 包含 text / chapter / section / chunk_id 等字段。
    """
    rules_path = os.path.join(DATA_RULES_DIR, 'rules_chunks.zh.json')
    if not os.path.exists(rules_path):
        print(f'[WARNING] 规则块文件不存在: {rules_path}')
        return []
    with open(rules_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f'加载 {len(chunks)} 个规则块')
    return chunks
