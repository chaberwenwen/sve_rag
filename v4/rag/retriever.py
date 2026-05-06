"""
向量检索模块 —— FAISS 双索引（card_names + content）加载与搜索

面试要点：
  - 单一职责：只负责「索引加载」和「向量搜索」，不关心卡牌识别和上下文
  - FAISS IndexFlatIP: 内积搜索，因向量已归一化所以等价于余弦相似度
  - 双库架构：
      card_names: 名称索引 → search_names() 用全文匹配识别卡牌
      content:    内容索引 → search_content() 用效果/问题文本搜语义相关QA+规则
  - v4 改进点：可替换为 IndexIVFFlat（加速）、HNSW（更高精度）、加入 Cross-Encoder 重排序
"""

import os
import pickle

import numpy as np
import faiss

from config import INDEX_DIR, TOP_K
from .embedder import encode_query


class Retriever:
    """
    向量检索器 —— 加载双 FAISS 索引并执行搜索。

    返回结果包含 score / text / metadata / index，供 context 模块使用。
    """

    def __init__(self):
        # 名称索引
        self.name_index = None
        self.name_metadatas = []
        self.name_texts = []

        # 内容索引
        self.content_index = None
        self.content_metadatas = []
        self.content_texts = []

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
        加载双 FAISS 索引：card_names + content。

        优先加载 card_names 和 content；
        不兼容旧版 combined/cards/rules 索引。
        """
        # 名称索引
        name_idx = os.path.join(INDEX_DIR, 'card_names.faiss')
        name_meta = os.path.join(INDEX_DIR, 'card_names_meta.pkl')
        name_texts_path = os.path.join(INDEX_DIR, 'card_names_texts.pkl')

        if os.path.exists(name_idx) and os.path.exists(name_meta):
            print('加载 card_names (名称索引)...')
            self.name_index = faiss.read_index(name_idx)
            with open(name_meta, 'rb') as f:
                self.name_metadatas = pickle.load(f)
            if os.path.exists(name_texts_path):
                with open(name_texts_path, 'rb') as f:
                    self.name_texts = pickle.load(f)
            print(f'  card_names: {self.name_index.ntotal} 条目')

        # 内容索引
        content_idx = os.path.join(INDEX_DIR, 'content.faiss')
        content_meta = os.path.join(INDEX_DIR, 'content_meta.pkl')
        content_texts_path = os.path.join(INDEX_DIR, 'content_texts.pkl')

        if os.path.exists(content_idx) and os.path.exists(content_meta):
            print('加载 content (内容索引)...')
            self.content_index = faiss.read_index(content_idx)
            with open(content_meta, 'rb') as f:
                self.content_metadatas = pickle.load(f)
            if os.path.exists(content_texts_path):
                with open(content_texts_path, 'rb') as f:
                    self.content_texts = pickle.load(f)
            print(f'  content: {self.content_index.ntotal} 条目')

        # 向后兼容：构建合并索引
        self._build_merged()

        if not self.name_index and not self.content_index:
            print('[WARNING] 未加载任何索引，请先运行: python -m v4.indexer.build')

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
                # 使用 reconstruct_n 提取向量（公共 API，兼容所有 faiss 版本）
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

        返回结果中 metadata 包含 cardno / name 等字段。
        """
        if self.name_index is None:
            return []
        return self._search_on(query, self.name_index,
                               self.name_metadatas, self.name_texts, top_k)

    # ============================================================
    # 内容检索
    # ============================================================
    def search_content(self, query: str, top_k: int = None) -> list[dict]:
        """
        在 content 索引中检索，用于效果/问题语义匹配 QA 和规则。
        """
        if self.content_index is None:
            return []
        return self._search_on(query, self.content_index,
                               self.content_metadatas, self.content_texts, top_k)

    # ============================================================
    # 向后兼容：统一检索（搜索合并索引）
    # ============================================================
    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        向量检索 top-K 相关文档（向后兼容，搜索合并索引）。
        """
        if self.index is None:
            return []
        return self._search_on(query, self.index,
                               self.metadatas, self.texts, top_k)

    # ============================================================
    # 通用搜索
    # ============================================================
    def _search_on(self, query: str, faiss_index,
                   metadatas: list, texts: list,
                   top_k: int = None) -> list[dict]:
        """对指定索引执行搜索，返回标准化结果列表。"""
        k = min(top_k or TOP_K, faiss_index.ntotal)

        query_vec = encode_query(query)
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
