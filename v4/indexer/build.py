"""
索引构建模块 —— 将卡牌/QA/规则数据编码为 FAISS 向量索引（双库多索引架构）

面试要点：
  - FAISS IndexFlatIP 使用内积度量，配合 L2 归一化的向量等价于余弦相似度
  - BGE 模型采用 attention-weighted mean pooling（与 BGE 官方一致）
  - 三索引架构：
      card_names  （名称索引）→ 原始 BGE 编码，用于全文匹配识别卡牌
      content     （内容索引）→ 原始 BGE 编码，LoRA 关闭时使用
      content_lora（内容索引）→ LoRA BGE 编码，LoRA 开启时使用
  - content / content_lora 使用完全相同的文本数据，仅编码器不同
  - 保存 text 原文供检索时直接输出，避免回查数据库

数据加载全部委托给 data/loader.py，本模块只负责「文本→向量→索引」的纯 ETL 流程。

注意：只索引纯数字编号的普通卡（已在 loader 层过滤），无需稀有度去重。
"""

import os
import pickle

import numpy as np

import config
from config import INDEX_DIR
from data.loader import load_cards, load_qa_cn, load_rules
from rag.embedder import encode_texts


# ============================================================
# 文本准备：名称索引
# ============================================================
def _prepare_name_texts(cards: list[dict]) -> tuple[list[str], list[dict]]:
    """
    名称索引：仅卡牌名称 + 编号 + 职业/类型 + 效果截断。

    用途：接收用户全文输入，在名称库中匹配识别提到的卡牌。
    每张卡牌生成 1 条索引记录。
    """
    texts = []
    metas = []

    for card in cards:
        cardno = card.get('cardno', '')
        name = card.get('name', '')
        desc = card.get('description', '')
        card_class = card.get('class', '')
        card_type = card.get('card_type', '')
        pack = card.get('pack', '')

        name_text = f'{name} ({cardno})'
        texts.append(name_text)
        metas.append({
            'type': 'card', 'cardno': cardno, 'name': name,
            'source': 'card_name', 'pack': pack,
        })

    print(f'  生成 {len(texts)} 条名称索引记录')
    return texts, metas


# ============================================================
# 文本准备：内容索引
# ============================================================
def _prepare_content_texts(qa_data: dict[str, list[dict]],
                            rules: list[dict]) -> tuple[list[str], list[dict]]:
    """
    内容索引：QA（无卡牌名前缀）+ 规则文本。

    关键改动：QA 条目不再以 '卡牌: {name} ({cardno})\n' 为前缀，
    避免卡牌名称污染嵌入向量，让相似度真正反映「效果/问题」的语义关联。

    用途：接收效果文本或用户问题，检索语义相关的 QA 和规则。
    """
    texts = []
    metas = []

    # QA 条目（去名前缀）
    for cardno, qas in qa_data.items():
        for qa in qas:
            q_text = qa.get('question_cn', '')
            a_text = qa.get('answer_cn', '')
            card_name = qa.get('card_name', '')
            if not q_text and not a_text:
                continue
            qa_text = f'问: {q_text}\n答: {a_text}'
            texts.append(qa_text)
            metas.append({
                'type': 'qa', 'cardno': cardno, 'name': card_name,
                'source': 'qa_cn',
            })

    # 规则条目
    for chunk in rules:
        texts.append(chunk['text'])
        metas.append({
            'type': 'rule',
            'chapter': chunk.get('chapter', ''),
            'section': chunk.get('section', ''),
            'source': 'rule',
            'chunk_id': chunk.get('chunk_id', ''),
        })

    print(f'  生成 {len(texts)} 条内容索引记录')
    return texts, metas


# ============================================================
# FAISS 索引构建
# ============================================================
def _build_one_index(texts: list[str], metas: list[dict], name: str,
                     use_lora: bool = False):
    """构建单个 FAISS 索引并保存到 index/ 目录。

    Args:
        use_lora: True=使用 LoRA BGE（content_lora），False=使用原始 BGE（card_names / content）
    """
    if not texts:
        print(f'  [跳过] {name}: 无文本')
        return

    os.makedirs(INDEX_DIR, exist_ok=True)

    model_tag = 'LoRA BGE' if use_lora else '原始 BGE'
    print(f'  编码 {len(texts)} 条文本 → 嵌入向量 [{model_tag}]...')
    embeddings = encode_texts(texts, normalize=True, use_lora=use_lora)

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度（因向量已归一化）
    index.add(embeddings.astype(np.float32))

    # 保存索引文件
    faiss.write_index(index, os.path.join(INDEX_DIR, f'{name}.faiss'))
    # 保存元数据
    with open(os.path.join(INDEX_DIR, f'{name}_meta.pkl'), 'wb') as f:
        pickle.dump(metas, f)
    # 保存原始文本（检索时直接输出，无需回查源文件）
    with open(os.path.join(INDEX_DIR, f'{name}_texts.pkl'), 'wb') as f:
        pickle.dump(texts, f)

    print(f'  {name}: {len(embeddings)} 条, 维度 {dim}, 已保存')


# ============================================================
# 主入口
# ============================================================
def build_all_indices():
    """
    构建三索引的主流程。

    card_names   → 原始 BGE（名称匹配）
    content      → 原始 BGE（LoRA 关闭时的内容检索）
    content_lora → LoRA BGE（LoRA 开启时的内容检索）

    content 和 content_lora 使用完全相同的文本数据。
    """
    print('=' * 50)
    print('开始构建向量索引 (三索引架构)')
    print('=' * 50)

    # -------- Step 1: 加载数据 --------
    print('\n[1/3] 加载数据...')
    cards = load_cards()
    qa_data = load_qa_cn()
    rules = load_rules()

    # -------- Step 2: 准备文本 --------
    print('\n[2/3] 准备索引文本...')
    name_texts, name_metas = _prepare_name_texts(cards)
    content_texts, content_metas = _prepare_content_texts(qa_data, rules)

    # -------- Step 3: 构建索引 --------
    print('\n[3/3] 构建 FAISS 索引...')

    print('\n--- 名称索引 (card_names) [原始BGE] ---')
    _build_one_index(name_texts, name_metas, 'card_names', use_lora=False)

    print('\n--- 内容索引 (content) [原始BGE] ---')
    _build_one_index(content_texts, content_metas, 'content', use_lora=False)

    # LoRA 内容索引（仅在 LoRA adapter 存在时构建）
    lora_dir = config.LORA_MODEL_DIR
    if os.path.exists(lora_dir):
        print('\n--- 内容索引 (content_lora) [LoRA BGE] ---')
        _build_one_index(content_texts, content_metas, 'content_lora', use_lora=True)
    else:
        print(f'\n[跳过] content_lora: LoRA adapter 不存在 ({lora_dir})')

    print('\n' + '=' * 50)
    print(f'索引构建完成 → {INDEX_DIR}')
    print('=' * 50)


def build_names_only():
    """仅构建名称索引（card_names），使用原始 BGE。"""
    print('=' * 50)
    print('仅构建名称索引 (card_names) [原始BGE]')
    print('=' * 50)

    print('\n加载卡牌数据...')
    cards = load_cards()

    print('准备名称索引文本...')
    name_texts, name_metas = _prepare_name_texts(cards)

    print('\n构建名称索引...')
    _build_one_index(name_texts, name_metas, 'card_names', use_lora=False)

    print(f'\n名称索引构建完成 → {INDEX_DIR}')


def build_content_only():
    """仅构建内容索引（content + content_lora）。"""
    print('=' * 50)
    print('仅构建内容索引 (content + content_lora)')
    print('=' * 50)

    print('\n加载 QA/规则数据...')
    qa_data = load_qa_cn()
    rules = load_rules()

    print('准备内容索引文本...')
    content_texts, content_metas = _prepare_content_texts(qa_data, rules)

    print('\n--- content [原始BGE] ---')
    _build_one_index(content_texts, content_metas, 'content', use_lora=False)

    lora_dir = config.LORA_MODEL_DIR
    if os.path.exists(lora_dir):
        print('\n--- content_lora [LoRA BGE] ---')
        _build_one_index(content_texts, content_metas, 'content_lora', use_lora=True)
    else:
        print(f'\n[跳过] content_lora: LoRA adapter 不存在 ({lora_dir})')

    print(f'\n内容索引构建完成 → {INDEX_DIR}')
