"""
线程安全 Embedder 封装 —— 解决 PyTorch 模型非线程安全问题。

原理：
  - PyTorch 的 Transformer 模型在推理时存在内部缓冲区竞争（attention mask 等）
  - 即使 CPU 推理也会有内存分配器的非线程安全问题
  - 通过 threading.Lock 将所有编码调用串行化，彻底避免竞争

FAISS 索引的 search() 是只读操作，天然线程安全，无需额外保护。
"""

import threading
from typing import Optional


class ThreadSafeEmbedder:
    """线程安全的 Embedding 编码器"""

    def __init__(self):
        self._lock = threading.Lock()
        self._base_embedder = None
        self._lora_embedder = None
        self._loaded = False

    def _ensure_loaded(self):
        """懒加载 embedder（首次编码时加载，避免启动时阻塞）"""
        if self._loaded:
            return
        from v4.rag.embedder import get_embedder

        self._base_embedder = get_embedder()
        self._loaded = True

    def encode_query(self, query: str, use_lora: bool = False):
        """编码单条查询文本（线程安全）"""
        self._ensure_loaded()
        with self._lock:
            from v4.rag.embedder import encode_query
            return encode_query(query, use_lora=use_lora)

    def encode_texts(self, texts: list[str], use_lora: bool = False):
        """编码批量文本（线程安全）"""
        self._ensure_loaded()
        with self._lock:
            from v4.rag.embedder import encode_texts
            return encode_texts(texts, use_lora=use_lora)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# 全局单例
_embedder: Optional[ThreadSafeEmbedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> ThreadSafeEmbedder:
    """获取全局线程安全 Embedder 单例"""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = ThreadSafeEmbedder()
    return _embedder
