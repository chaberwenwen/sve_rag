"""
GET /api/health —— 健康检查接口。

返回服务状态，包括 Embedder 加载状态、FAISS 索引状态、LLM 连通性。
"""

import os
from fastapi import APIRouter, Depends
from server.config import settings
from server.schemas import HealthResponse, FaissIndicesStatus
from server.model_pool import get_embedder
from server.llm_async import get_llm_client, AsyncLLMClient

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health_check(
    llm_client: AsyncLLMClient = Depends(get_llm_client),
):
    """
    健康检查：返回各组件状态。

    - embedder_loaded: Embedding 模型是否已加载
    - faiss_indices: 三个 FAISS 索引是否存在
    - llm_enabled: LLM 全局开关状态
    - llm_reachable: LLM API 是否可达
    """
    embedder = get_embedder()

    # 检查 FAISS 索引文件
    index_dir = settings.index_dir
    faiss_status = FaissIndicesStatus(
        card_names=os.path.exists(os.path.join(index_dir, "card_names.faiss")),
        content=os.path.exists(os.path.join(index_dir, "content.faiss")),
        content_lora=os.path.exists(os.path.join(index_dir, "content_lora.faiss")),
    )

    # 检查 LLM 连通性
    llm_reachable = False
    if settings.llm_enabled:
        try:
            llm_reachable = await llm_client.health_check()
        except Exception:
            llm_reachable = False

    return HealthResponse(
        status="ok",
        embedder_loaded=embedder.is_loaded,
        faiss_indices=faiss_status,
        llm_enabled=settings.llm_enabled,
        llm_provider=settings.llm_provider,
        llm_reachable=llm_reachable,
    )
