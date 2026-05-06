"""
上下文构建模块 —— 将卡牌/QA/检索结果组装为 LLM 输入

面试要点：
  - 单一职责：只负责「如何组织上下文文本」，不关心检索和推理
  - 上下文按「卡牌信息 → 检索结果」组织，让 LLM 优先理解卡牌再结合规则
  - QA 统一由内容索引向量召回提供，不在卡牌信息区全量列出
  - v4 改进点：可加入长度截断策略、滑动窗口、多轮对话历史等
"""

from .card_matcher import CardMatcher


class ContextBuilder:
    """
    上下文构建器 —— 将卡牌信息、向量检索结果组装为结构化的 LLM 上下文。
    QA 不再从 card_matcher 全量获取，而是依赖内容索引的向量召回。
    """

    def __init__(self, card_matcher: CardMatcher):
        self.card_matcher = card_matcher

    def build(self, cards: list[dict], search_results: list[dict]) -> str:
        """
        构建 LLM 上下文。

        组织顺序：
          1. 卡牌信息（名称/编号/职业/类型/费用/攻/命/效果）
          2. 检索到的规则/QA 文档（含向量召回的相关 QA）
        """
        parts = []

        # --- 卡牌信息 ---
        if cards:
            parts.append('=== 卡牌信息 ===')
            for card in cards:
                cardno = card.get('cardno', '')
                name = card.get('name', '')
                desc = card.get('description', '')
                card_type = card.get('card_type', '')
                card_class = card.get('class', '')
                cost = card.get('cost', '')
                power = card.get('power', '')
                hp = card.get('hp', '')

                parts.append(f'卡牌: {name} ({cardno})')
                parts.append(f'职业: {card_class} | 类型: {card_type} | 费用: {cost} | 攻: {power} | 命: {hp}')
                parts.append(f'效果: {desc}')
                parts.append('')

        # --- 检索结果 ---
        if search_results:
            parts.append('=== 检索到的相关规则和 QA ===')
            for r in search_results:
                text = r.get('text', '')
                meta = r['metadata']
                source = meta.get('source', '')

                if source == 'rule':
                    label = f'[规则] Ch.{meta.get("chapter", "")} Sec.{meta.get("section", "")}'
                elif source == 'qa_cn':
                    label = f'[QA] {meta.get("name", "")} ({meta.get("cardno", "")})'
                elif source in ('card_effect', 'card_name'):
                    label = f'[卡牌] {meta.get("name", "")} ({meta.get("cardno", "")})'
                else:
                    label = '[未知来源]'

                parts.append(label)
                if text:
                    parts.append(text)
                parts.append('---')

        return '\n'.join(parts)
