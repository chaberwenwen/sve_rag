"""
向量检索模块 —— FAISS 三索引（card_names + content + content_lora）加载与搜索

面试要点：
  - 单一职责：只负责「索引加载」和「向量搜索」，不关心卡牌识别和上下文
  - FAISS IndexFlatIP: 内积搜索，因向量已归一化所以等价于余弦相似度
  - 三索引架构：
      card_names   名称索引 → 原始 BGE（全文匹配识别卡牌，始终不变）
      content      内容索引 → 原始 BGE（LoRA 关闭时使用）
      content_lora 内容索引 → LoRA BGE（LoRA 开启时使用）
  - 运行时根据 config.LORA_ENABLED 自动选择 content / content_lora
  - 编码器与索引始终保持一致，无向量空间不匹配问题
"""

import os
import pickle

import numpy as np
import faiss

import config
from config import INDEX_DIR, TOP_K
from .embedder import encode_query


class Retriever:
    """
    向量检索器 —— 加载三 FAISS 索引并执行搜索。

    返回结果包含 score / text / metadata / index，供 context 模块使用。
    """

    def __init__(self):
        # 名称索引
        self.name_index = None
        self.name_metadatas = []
        self.name_texts = []

        # 内容索引（原始 BGE）
        self.content_index = None
        self.content_metadatas = []
        self.content_texts = []

        # 内容索引（LoRA BGE）
        self.content_lora_index = None
        self.content_lora_metadatas = []
        self.content_lora_texts = []

        # 向后兼容：合并索引（供旧代码用 combined 方式访问）
        self.index = None
        self.metadatas = []
        self.texts = []

        self._load_indices()

    # ============================================================
    # 索引加载
    # ============================================================
    def _load_indices(self):
        """
        加载三 FAISS 索引：card_names + content + content_lora。

        content 和 content_lora 使用完全相同的元数据和文本，
        但向量空间不同（原始 BGE vs LoRA BGE）。
        """

        def _load_one(prefix):
            """加载单个索引及元数据，返回 (index, metas, texts)。"""
            idx_path = os.path.join(INDEX_DIR, f'{prefix}.faiss')
            meta_path = os.path.join(INDEX_DIR, f'{prefix}_meta.pkl')
            texts_path = os.path.join(INDEX_DIR, f'{prefix}_texts.pkl')

            if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
                return None, [], []

            index = faiss.read_index(idx_path)
            with open(meta_path, 'rb') as f:
                metas = pickle.load(f)
            texts = []
            if os.path.exists(texts_path):
                with open(texts_path, 'rb') as f:
                    texts = pickle.load(f)

            print(f'  {prefix}: {index.ntotal} 条目')
            return index, metas, texts

        # 名称索引
        print('加载 card_names (名称索引)...')
        self.name_index, self.name_metadatas, self.name_texts = _load_one('card_names')

        # 内容索引（原始 BGE）
        print('加载 content (内容索引·原始BGE)...')
        self.content_index, self.content_metadatas, self.content_texts = _load_one('content')

        # 内容索引（LoRA BGE）
        print('加载 content_lora (内容索引·LoRA BGE)...')
        self.content_lora_index, self.content_lora_metadatas, self.content_lora_texts = _load_one('content_lora')

        # 向后兼容：构建合并索引
        self._build_merged()

        if not self.name_index and not self.content_index and not self.content_lora_index:
            print('[WARNING] 未加载任何索引，请先运行: python v4/run.py index')

    def _build_merged(self):
        """向后兼容：将 card_names + content 合并为一个索引。"""
        indices = []
        for idx, meta_list, text_list in [
            (self.name_index, self.name_metadatas, self.name_texts),
            (self.content_index, self.content_metadatas, self.content_texts),
        ]:
            if idx is not None:
                indices.append(idx)
                self.metadatas.extend(meta_list)
                self.texts.extend(text_list)

        if len(indices) == 0:
            return

        if len(indices) == 1:
            self.index = indices[0]
        else:
            all_vecs = []
            for idx in indices:
                dim = idx.d
                n = idx.ntotal
                vecs = idx.reconstruct_n(0, n)
                all_vecs.append(vecs)
            all_vecs = np.vstack(all_vecs)
            self.index = faiss.IndexFlatIP(all_vecs.shape[1])
            self.index.add(all_vecs.astype(np.float32))

        print(f'  合并索引: {self.index.ntotal} 条目 (向后兼容)')

    # ============================================================
    # 名称检索
    # ============================================================
    def search_names(self, query: str, top_k: int = None) -> list[dict]:
        """
        在 card_names 索引中检索，用于通过全文匹配识别卡牌。

        始终使用原始 BGE 编码（名称索引始终由原始 BGE 构建）。
        返回结果中 metadata 包含 cardno / name 等字段。
        """
        if self.name_index is None:
            return []
        return self._search_on(query, self.name_index,
                                self.name_metadatas, self.name_texts,
                                top_k, use_lora=False)

    # ============================================================
    # 内容检索（运行时自动选择 content / content_lora）
    # ============================================================
    def _get_active_content(self):
        """根据 config.LORA_ENABLED 返回当前活跃的内容索引及编码模式。"""
        if config.LORA_ENABLED and self.content_lora_index is not None:
            return self.content_lora_index, self.content_lora_metadatas, self.content_lora_texts, True
        elif self.content_index is not None:
            return self.content_index, self.content_metadatas, self.content_texts, False
        elif self.content_lora_index is not None:
            # 回退：content 不存在但 content_lora 存在
            return self.content_lora_index, self.content_lora_metadatas, self.content_lora_texts, True
        else:
            return None, [], [], False

    def search_content(self, query: str, top_k: int = None,
                       use_lora: bool = None) -> list[dict]:
        """
        在内容索引中检索，用于效果/问题语义匹配 QA 和规则。

        运行时自动选择：
          - config.LORA_ENABLED=True  → content_lora（LoRA BGE 编码）
          - config.LORA_ENABLED=False → content（原始 BGE 编码）

        Args:
            use_lora: 显式指定编码模式。为 None 时使用 config.LORA_ENABLED 自动选择。
                      显式传入 True/False 时覆盖默认行为。
        """
        if use_lora is not None:
            # 显式指定模式
            if use_lora and self.content_lora_index is not None:
                idx, metas, texts = self.content_lora_index, self.content_lora_metadatas, self.content_lora_texts
            elif self.content_index is not None:
                idx, metas, texts = self.content_index, self.content_metadatas, self.content_texts
            elif self.content_lora_index is not None:
                idx, metas, texts = self.content_lora_index, self.content_lora_metadatas, self.content_lora_texts
                use_lora = True
            else:
                return []
        else:
            idx, metas, texts, use_lora = self._get_active_content()
        if idx is None:
            return []
        return self._search_on(query, idx, metas, texts, top_k, use_lora=use_lora)

    # ============================================================
    # 向后兼容：统一检索（搜索合并索引）
    # ============================================================
    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        向量检索 top-K 相关文档（向后兼容，搜索合并索引）。
        始终使用原始 BGE 编码。
        """
        if self.index is None:
            return []
        return self._search_on(query, self.index,
                                self.metadatas, self.texts, top_k,
                                use_lora=False)

    # ============================================================
    # 通用搜索
    # ============================================================
    def _search_on(self, query: str, faiss_index,
                   metadatas: list, texts: list,
                   top_k: int = None,
                   use_lora: bool = False) -> list[dict]:
        """对指定索引执行搜索，返回标准化结果列表。

        Args:
            use_lora: True=使用 LoRA BGE 编码（content_lora），False=使用原始 BGE（其他）
        """
        k = min(top_k or TOP_K, faiss_index.ntotal)

        query_vec = encode_query(query, use_lora=use_lora)
        distances, indices = faiss_index.search(query_vec.astype(np.float32), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(metadatas):
                continue
            text = texts[idx] if idx < len(texts) else ''
            results.append({
                'score': float(distances[0][i]),
                'text': text,
                'metadata': metadatas[idx],
                'index': int(idx),
            })
        return results
