"""
卡牌识别模块 —— 从用户输入提取/匹配卡牌

面试要点：
  - 单一职责：只负责「从文本中找到卡牌」，不关心向量检索和上下文构建
  - 4 级匹配策略：精确匹配 → 正则提取编号 → 子串双向匹配 → 去重
  - card_map 建立多重索引（编号/名称/大小写），加速查找
  - 只索引纯数字编号的普通卡（已在 loader 层过滤），无需 variant_to_base
  - v4 改进点：可加入模糊匹配（fuzzywuzzy）、同义词映射等
"""

import re

from data.loader import load_cards, load_qa_cn

# 用于拆分子串匹配的中文连词和分隔符
_SPLIT_PATTERN = re.compile(r'[\s,，、和与及跟同\|\/]+')


class CardMatcher:
    """
    卡牌识别器 —— 从用户输入中识别卡牌，提供关联 QA 查询。

    使用 card_map 多重索引加速查找：
      - cardno (大写)        → 精确编号匹配
      - cardno (小写)        → 小写编号匹配
      - name    (原样)       → 中文名称匹配
      - name    (小写)       → 小写名称匹配
    """

    def __init__(self):
        self.card_map = {}   # key → card 的多重映射
        self.qa_data = {}    # cardno → list[dict]
        self._build_card_map()

    # ============================================================
    # 构建多重索引
    # ============================================================
    def _build_card_map(self):
        """加载卡牌和 QA 数据，构建多重索引。

        编号映射：始终记录（同一 cardno 只出现一次）
        名称映射：同名卡优先选基础形态（cost >= 0），与 QA 映射策略一致
        """
        cards = load_cards()

        for card in cards:
            cardno = card.get('cardno', '')
            name = card.get('name', '')
            cost = card.get('cost', 0) or 0

            # 编号映射：每个 cardno 只出现一次
            if cardno not in self.card_map:
                self.card_map[cardno] = card
                self.card_map[cardno.lower()] = card

            # 名称映射：同名卡优先选基础形态（cost >= 0）
            if name:
                existing = self.card_map.get(name)
                if existing is None or (cost >= 0 and (existing.get('cost', 0) or 0) < 0):
                    self.card_map[name] = card
                    self.card_map[name.lower()] = card

        # 加载 QA
        self.qa_data = load_qa_cn()

        loaded = len(set(id(c) for c in self.card_map.values()))
        print(f'card_matcher: {len(self.card_map)} 个 key → {loaded} 张独立卡牌')

    # ============================================================
    # 卡牌识别（4 级匹配）
    # ============================================================
    def find_cards(self, user_input: str) -> list[dict]:
        """
        从用户输入中识别卡牌，返回匹配的卡牌数据列表。

        识别策略（按优先级）：
          1. 精确匹配输入本身（编号或名称）
          2. 正则提取 card_no 格式的编号（如 BP01-001）
          3. 名称子串双向匹配
             - 输入子串包含在卡名中（如输入"桃乐丝" → "次元魔女·桃乐丝"）
             - 卡名包含在输入中（如输入"玫瑰皇后的效果" → "玫瑰皇后"）
        """
        if not user_input or not self.card_map:
            return []

        input_lower = user_input.lower().strip()
        found = []

        # --- 1) 精确匹配 ---
        if input_lower in self.card_map:
            return [self.card_map[input_lower]]

        # --- 2) 正则提取编号 ---
        pattern = re.compile(r'([A-Z]{2}\d+[-][A-Z0-9]+)', re.IGNORECASE)
        for m in pattern.findall(user_input):
            key = m.upper()
            if key in self.card_map:
                found.append(self.card_map[key])

        if found:
            return _deduplicate(found)

        # --- 3) 名称子串匹配 ---
        # 用户可能一次问多张卡，如"次元超越和桃乐丝"
        # 先按分隔符拆成片段，对每个片段做双向子串匹配
        fragments = [f.strip() for f in _SPLIT_PATTERN.split(input_lower) if len(f.strip()) >= 2]
        if not fragments:
            fragments = [input_lower] if len(input_lower) >= 2 else []

        for fragment in fragments:
            for key, card in self.card_map.items():
                card_name = card.get('name', '').lower()
                if not card_name:
                    continue
                if fragment in card_name or card_name in fragment:
                    found.append(card)

        return _deduplicate(found)

    # ============================================================
    # QA 查询
    # ============================================================
    def get_qa(self, cardno: str) -> list[dict]:
        """
        获取指定卡牌的中文 QA。

        优先按 cardno 精确查询；若 cardno 无 QA（如进化卡），
        则通过卡牌名称回退查找基础卡的 QA。
        """
        qas = self.qa_data.get(cardno)
        if qas:
            return qas

        # cardno 无 QA，尝试通过卡牌名称查找
        card = self.card_map.get(cardno)
        if card:
            name = card.get('name', '')
            if name:
                # 通过名称搜索同名基础卡（可能有 QA）
                for key, c in self.card_map.items():
                    if c.get('name', '') == name and key in self.qa_data:
                        return self.qa_data[key]
        return []


def _deduplicate(cards: list[dict]) -> list[dict]:
    """按 cardno 去重，保持插入顺序。"""
    seen = set()
    result = []
    for c in cards:
        cno = c.get('cardno', '')
        if cno not in seen:
            seen.add(cno)
            result.append(c)
    return result
