"""
RAG Pipeline —— 检索 + LLM 生成的总编排

面试要点：
  - Pipeline 是「检索-增强-生成」的核心编排层
  - query() 方法拆分为 6 步：名称索引识别卡牌 → 提取效果 → 内容双向量检索 → 收集QA → 构建上下文 → 调用LLM
  - identify_cards() 单独暴露卡牌识别步骤，供 CLI/UI 做手动卡牌选择
  - query() 支持 selected_cardnos 参数，允许用户筛选卡牌后再检索
  - 每步可单独替换/扩展（如更换检索策略或 LLM 后端）
  - 上下文构建时按「卡牌信息 → 检索结果」组织，QA 统一由内容索引向量召回
  - _resolve_cards_from_name_hits 通过 CardMatcher.card_map 补全卡牌完整数据

双库检索流程：
  1. 用户输入全文 → card_names 索引 → 识别提到的卡牌
  2. （可选）用户手动选择卡牌，过滤掉不相关的噪音卡
  3. 提取匹配卡牌的效果文本作为 effect_query
  4. 内容索引用 effect_query + 原始输入双向量检索，合并去重取 Top-K

职责边界：
  - 不处理数据加载（委托 data/loader.py）
  - 不处理索引构建（委托 indexer/）
  - 只做「编排」：把各组件串起来
"""

from config import TOP_K, SYSTEM_PROMPT, NAME_SCORE_RATIO_THRESHOLD, \
    CONTENT_SCORE_MIN_THRESHOLD
from .card_matcher import CardMatcher
from .retriever import Retriever
from .context import ContextBuilder
from .llm_client import LLMClient


def _merge_results(results_a: list[dict], results_b: list[dict],
                   top_k: int,
                   score_threshold: float = None) -> list[dict]:
    """合并两组检索结果，按 index 去重保留最高分，过滤低分噪声，取 Top-K。

    两层过滤策略：
      1. 绝对阈值过滤：丢弃分数 < CONTENT_SCORE_MIN_THRESHOLD 的结果
      2. 相对比值过滤：丢弃分数 < 最高分 × CONTENT_SCORE_RATIO_THRESHOLD 的结果

    两重条件取交集（二者都满足才保留），能有效过滤 LoRA 训练偏差产生的中等分噪声。

    Args:
        results_a: 效果文本检索结果（LoRA 训练过的路径，通常质量较高）
        results_b: 用户原始问题检索结果（LoRA 未训练过的路径，可能全是噪声）
        top_k: 最终返回数量
        score_threshold: 最低绝对分数阈值，默认使用 CONTENT_SCORE_MIN_THRESHOLD
    """
    from config import CONTENT_SCORE_RATIO_THRESHOLD

    abs_threshold = score_threshold if score_threshold is not None \
        else CONTENT_SCORE_MIN_THRESHOLD

    # 先合并（取最高分）
    merged = {}
    for r in results_a + results_b:
        idx = r['index']
        if idx not in merged or r['score'] > merged[idx]['score']:
            merged[idx] = r

    if not merged:
        return []

    # 计算相对阈值：最高分 × RATIO
    max_score = max(r['score'] for r in merged.values())
    rel_threshold = max_score * CONTENT_SCORE_RATIO_THRESHOLD

    # 双层过滤：绝对阈值 AND 相对阈值
    filtered = []
    for r in merged.values():
        if abs_threshold > 0 and r['score'] < abs_threshold:
            continue  # 低于绝对阈值 → 丢弃
        if rel_threshold > 0 and r['score'] < rel_threshold:
            continue  # 低于相对阈值 → 丢弃
        filtered.append(r)

    sorted_results = sorted(filtered,
                            key=lambda x: x['score'], reverse=True)
    return sorted_results[:top_k]


class RAGPipeline:
    """RAG 检索-生成流水线"""

    def __init__(self, base_url: str = None, model: str = None):
        self.card_matcher = CardMatcher()
        self.retriever = Retriever()
        self.context_builder = ContextBuilder(self.card_matcher)
        self.llm = LLMClient(base_url, model)

    # ============================================================
    # 公开 API
    # ============================================================

    def identify_cards(self, user_input: str) -> list[dict]:
        """
        仅执行卡牌识别步骤，返回候选卡牌列表（供手动选择）。

        这是 query() 中 Step 1 的独立版本，用于 CLI/UI 展示识别结果
        并让用户选择后再调用 query() 进行检索。

        Returns:
            list[dict]: 候选卡牌列表，每项包含 cardno, name, description, score 等
        """
        name_hits = self.retriever.search_names(user_input, top_k=5)

        # 分数阈值过滤
        if name_hits and NAME_SCORE_RATIO_THRESHOLD > 0:
            top_score = name_hits[0]['score']
            min_score = top_score * NAME_SCORE_RATIO_THRESHOLD
            name_hits = [h for h in name_hits if h['score'] >= min_score]

        found_cards = self._resolve_cards_from_name_hits(name_hits)

        # 同时保留原 CardMatcher 的精确匹配作为补充
        exact_cards = self.card_matcher.find_cards(user_input)
        found_cards = self._merge_cards(found_cards, exact_cards)

        return found_cards

    def query(self, user_input: str,
              selected_cardnos: list[str] = None,
              use_llm: bool = True,
              use_lora: bool = False) -> dict:
        """
        执行一次完整的 RAG 查询。

        Args:
            user_input: 用户输入的问题
            selected_cardnos: 可选，用户手动选择的卡牌编号列表。
                              为 None 时使用自动识别的全部卡牌。
                              为空列表 [] 时跳过卡牌识别，直接内容检索。
            use_llm: 是否调用 LLM 生成回答。False 时仅返回检索结果。

        Steps:
          1. 名称索引：用全文匹配识别卡牌（或用 selected_cardnos）
          2. 提取匹配卡牌的效果文本
          3a. 内容索引：效果文本检索
          3b. 内容索引：原始问题检索
          3c. 合并去重取 Top-K
          4. 收集已识别卡牌的关联 QA
          5. 构建 LLM 上下文
          6. （可选）调用 LLM 生成回答

        Returns:
            {
                'answer': str,           # LLM 生成的回答（LLM 关闭时为 None）
                'cards': list[dict],     # 识别的卡牌（名称索引）
                'search_results': list,  # top-K 检索结果（内容索引合并）
                'qa_results': list,      # 关联的 QA
                'context': str,          # 构建的上下文
                'prompt': str,           # 完整 prompt（含 system）
            }
        """
        # Step 1: 名称索引 → 识别卡牌
        if selected_cardnos is not None:
            # 用户手动选择卡牌：从 card_map 中按 cardno 取
            found_cards = []
            for cardno in selected_cardnos:
                full = self.card_matcher.card_map.get(cardno)
                if full:
                    found_cards.append({
                        'cardno': cardno,
                        'name': full.get('name', ''),
                        'description': full.get('description', ''),
                        'card_type': full.get('card_type', ''),
                        'class': full.get('class', ''),
                        'cost': full.get('cost', ''),
                        'power': full.get('power', ''),
                        'hp': full.get('hp', ''),
                        'score': 1.0,  # 手动选择的卡牌，分数为 1.0
                    })
        else:
            # 自动识别模式
            found_cards = self.identify_cards(user_input)

        # Step 2: 提取效果文本
        effect_texts = []
        for card in found_cards:
            desc = card.get('description', '').strip()
            if desc:
                effect_texts.append(desc)
        effect_query = '\n'.join(effect_texts) if effect_texts else ''

        # Step 3a/3b: 内容索引 — 双向量检索
        if effect_query:
            results_effect = self.retriever.search_content(effect_query, top_k=TOP_K, use_lora=use_lora)
        else:
            results_effect = []
        results_question = self.retriever.search_content(user_input, top_k=TOP_K, use_lora=use_lora)

        # Step 3c: 合并去重
        # LoRA 模式下禁用绝对阈值（分数整体偏低），仅依赖相对比值过滤
        _abs_threshold = 0.0 if use_lora else None  # None → 使用默认 CONTENT_SCORE_MIN_THRESHOLD
        search_results = _merge_results(results_effect, results_question, TOP_K,
                                        score_threshold=_abs_threshold)

        # Step 4: 收集关联 QA
        qa_results = []
        for card in found_cards:
            cardno = card.get('cardno', '')
            for qa in self.card_matcher.get_qa(cardno):
                qa_results.append({
                    'cardno': cardno,
                    'name': card.get('name', ''),
                    'qa': qa,
                })

        # Step 4b: 将 qa_results 合并到 search_results（去重后作为上下文）
        search_results = self._merge_qa_results(search_results, qa_results)

        # Step 5: 构建上下文
        context = self.context_builder.build(found_cards, search_results)

        # Step 6: 调用 LLM（可选）
        card_names = ', '.join([c.get('name', '') for c in found_cards])

        user_prompt = (
            f'用户输入: {user_input}\n'
            f'识别的卡牌: {card_names if card_names else "（未识别到具体卡牌）"}\n\n'
            f'以下是相关的卡牌信息和规则:\n{context}\n\n'
            f'请根据以上信息回答用户的问题。\n'
            f'回答要求：\n'
            f'1. 从上方检索结果中选出与问题最直接相关的 1~3 条 [QA] 或 [规则] 作为依据；\n'
            f'2. 引用时明确指出卡牌名称和具体规则内容，并解释为什么适用于当前问题；\n'
            f'3. 优先使用检索分数最高的结果，但若最高分结果明显不相关则跳过。'
        )

        if use_llm:
            answer = self.llm.call(user_prompt, SYSTEM_PROMPT)
        else:
            answer = None

        # 构建完整 prompt 文本（方便调试 LLM）
        full_prompt = f'[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}'

        return {
            'answer': answer,
            'cards': found_cards,
            'search_results': search_results,
            'qa_results': qa_results,
            'context': context,
            'prompt': full_prompt,
        }

    # ============================================================
    # 名称检索结果 → 卡牌数据
    # ============================================================
    def _resolve_cards_from_name_hits(self,
                                       name_hits: list[dict]) -> list[dict]:
        """从名称索引的检索结果中提取卡牌数据，通过 card_map 补全信息。"""
        cards = []
        seen = set()
        for hit in name_hits:
            meta = hit.get('metadata', {})
            cardno = meta.get('cardno', '')
            if cardno and cardno not in seen:
                seen.add(cardno)
                # 优先从 CardMatcher 的 card_map 获取完整卡牌数据
                full = self.card_matcher.card_map.get(cardno)
                if full:
                    cards.append({
                        'cardno': cardno,
                        'name': full.get('name', meta.get('name', '')),
                        'description': full.get('description', ''),
                        'card_type': full.get('card_type', ''),
                        'class': full.get('class', ''),
                        'cost': full.get('cost', ''),
                        'power': full.get('power', ''),
                        'hp': full.get('hp', ''),
                        'score': hit.get('score', 0.0),
                    })
                else:
                    cards.append({
                        'cardno': cardno,
                        'name': meta.get('name', ''),
                        'description': '',
                        'score': hit.get('score', 0.0),
                    })
        return cards

    @staticmethod
    def _merge_cards(name_cards: list[dict],
                     exact_cards: list[dict]) -> list[dict]:
        """合并名称索引匹配和精确匹配的卡牌列表，按 cardno 去重。"""
        merged = {c['cardno']: c for c in name_cards}
        for card in exact_cards:
            cardno = card.get('cardno', '')
            if cardno and cardno not in merged:
                merged[cardno] = card
            elif cardno:
                # 精确匹配的优先，因为可能有更完整的 description
                merged[cardno] = card
        return list(merged.values())

    @staticmethod
    def _merge_qa_results(search_results: list[dict],
                          qa_results: list[dict]) -> list[dict]:
        """
        将 CardMatcher 关联的 QA 合并到向量检索结果中，去重后返回。

        去重策略：
          - 以 (cardno, question_cn) 作为唯一键
          - 向量检索结果（search_results）中的 QA 已在内容索引中，
            格式为 '问: {q_text}\\n答: {a_text}'，读取 text 提取问题部分
          - qa_results 中的 QA 来自 card_matcher.get_qa()，
            格式为 {'question_cn': ..., 'answer_cn': ...}
          - 向量检索结果优先保留（有相似度分数，更可靠）
        """
        # 从现有 search_results 中提取已有 QA 的 (cardno, question) 集合
        existing_keys: set[tuple[str, str]] = set()
        for r in search_results:
            meta = r.get('metadata', {})
            if meta.get('source') == 'qa_cn':
                cardno = meta.get('cardno', '')
                text = r.get('text', '')
                # 提取问句部分：'问: xxx\\n答: ...' → 'xxx'
                question = ''
                if text.startswith('问: '):
                    q_end = text.find('\n答:')
                    if q_end > 0:
                        question = text[3:q_end]
                existing_keys.add((cardno, question))

        # 将未覆盖的 qa_results 转换为 search_results 格式
        for qa_entry in qa_results:
            qa = qa_entry.get('qa', {})
            q_text = qa.get('question_cn', '')
            a_text = qa.get('answer_cn', '')
            cardno = qa_entry.get('cardno', '')
            card_name = qa_entry.get('name', '')

            if not q_text and not a_text:
                continue

            dedup_key = (cardno, q_text)
            if dedup_key in existing_keys:
                continue  # 已存在，跳过
            existing_keys.add(dedup_key)

            qa_text = f'问: {q_text}\n答: {a_text}'
            search_results.append({
                'text': qa_text,
                'metadata': {
                    'type': 'qa',
                    'cardno': cardno,
                    'name': card_name,
                    'source': 'qa_cn',
                },
                'score': 0.0,
                'index': -1,
            })

        return search_results
