"""
POST /api/identify —— 卡牌识别接口。
"""

from fastapi import APIRouter, Depends
from server.schemas import IdentifyRequest, IdentifyResponse, CardCandidate
from server.dependencies import get_rag_pipeline

router = APIRouter()


@router.post("/api/identify", response_model=IdentifyResponse)
async def identify_cards(
    req: IdentifyRequest,
    pipeline = Depends(get_rag_pipeline),
):
    """识别输入中提到的卡牌，返回候选列表"""
    candidates = pipeline.identify_cards(req.user_input)

    return IdentifyResponse(
        candidates=[
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
            for c in candidates
        ]
    )
