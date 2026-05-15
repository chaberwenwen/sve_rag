"""
FastAPI 依赖注入 —— 管理 RAG Pipeline 和外部资源的生命周期。

所有依赖通过 Depends() 注入到路由中，便于测试和替换。
"""

import sys
import os
from functools import lru_cache

# 将项目根目录和 v4/ 目录加入 sys.path
# v4/ 需要在路径中，因为 v4/rag/embedder.py 使用 from config import ...
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_project_root, os.path.join(_project_root, 'v4')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server.config import settings
from server.model_pool import get_embedder, ThreadSafeEmbedder
from server.llm_async import get_llm_client, AsyncLLMClient


# ============================================================
# RAG Pipeline（懒加载 + 缓存）
# ============================================================

_pipeline = None
_pipeline_lock = None


def _get_pipeline_lock():
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading
        _pipeline_lock = threading.Lock()
    return _pipeline_lock


def get_pipeline():
    """
    获取全局 RAGPipeline 实例（线程安全懒加载）。

    注：pipeline 内部的 Retriever + CardMatcher 读取的都是只读 FAISS 索引，
    只有 LLM 调用需要线程安全保护（通过 AsyncLLMClient 的 Semaphore 实现），
    Embedder 编码通过 ThreadSafeEmbedder 的 Lock 保护。
    """
    global _pipeline
    if _pipeline is None:
        with _get_pipeline_lock():
            if _pipeline is None:
                from v4.rag.pipeline import RAGPipeline
                _pipeline = RAGPipeline()
    return _pipeline


# ============================================================
# 可注入的依赖（用于 FastAPI Depends）
# ============================================================

async def get_settings():
    """注入服务端配置"""
    return settings


async def get_rag_pipeline():
    """注入 RAG Pipeline"""
    return get_pipeline()


async def get_async_llm() -> AsyncLLMClient:
    """注入异步 LLM 客户端"""
    return get_llm_client()


async def get_safe_embedder() -> ThreadSafeEmbedder:
    """注入线程安全 Embedder"""
    return get_embedder()
