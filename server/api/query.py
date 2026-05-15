"""
POST /api/query —— RAG 查询接口。

支持 enable_llm 开关：关闭时仅返回向量检索结果，不调用 LLM。
"""

from fastapi import APIRouter, Depends
from server.config import settings
from server.schemas import (
    QueryRequest,
    QueryResponse,
    CardCandidate,
    QAResult,
    SearchResultItem,
)
from server.dependencies import get_rag_pipeline
from server.llm_async import AsyncLLMClient, get_llm_client

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
async def rag_query(
    req: QueryRequest,
    pipeline = Depends(get_rag_pipeline),
    llm_client: AsyncLLMClient = Depends(get_llm_client),
):
    """
    RAG 查询：根据用户选择和问题，执行完整的检索-生成流程。

    - enable_llm=true + settings.llm_enabled=true: 完整 RAG：向量检索 + LLM 生成回答
    - 否则: 仅向量检索，不调用 LLM（pipeline 内置同步 LLM 也会被跳过）
    """
    # 决定是否使用 LLM（前端开关 AND 全局开关）
    should_use_llm = settings.llm_enabled and req.enable_llm

    # selected_cardnos 为空列表时传 None，使用自动识别模式
    selected = req.selected_cardnos if req.selected_cardnos else None

    # 执行向量检索（Steps 1-5），LLM 调用在 pipeline 内部根据 use_llm 决定
    result = pipeline.query(req.user_input, selected, use_llm=should_use_llm, use_lora=req.use_lora)

    # 构建响应
    return QueryResponse(
        answer=result.get('answer'),
        matched_cards=[
            CardCandidate(
                cardno=c.get("cardno", ""),
                name=c.get("name", ""),
                description=c.get("description", ""),
                card_type=c.get("card_type", ""),
                class_=c.get("class", ""),
                cost=str(c.get("cost", "")),
                power=str(c.get("power", "")),
                hp=str(c.get("hp", "")),
                score=c.get("score", 0.0),
            )
            for c in result.get("cards", [])
        ],
        qa_results=[
            QAResult(
                cardno=qa.get("cardno", ""),
                name=qa.get("name", ""),
                qa=qa.get("qa", {}),
            )
            for qa in result.get("qa_results", [])
        ],
        search_results=[
            SearchResultItem(
                id=str(r.get("index", "")),
                text=r.get("text", ""),
                score=float(r.get("score", 0.0) or 0.0),
                source=r.get("metadata", {}).get("source", ""),
                cardno=r.get("metadata", {}).get("cardno", ""),
                card_name=r.get("metadata", {}).get("name", ""),
            )
            for r in result.get("search_results", [])
        ],
        context=result.get("context", ""),
        prompt=result.get("prompt", ""),
    )
