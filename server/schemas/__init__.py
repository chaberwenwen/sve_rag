"""
Pydantic 请求/响应模型定义。
"""

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 请求模型
# ============================================================

class IdentifyRequest(BaseModel):
    """卡牌识别请求"""
    user_input: str = Field(
        ...,
        max_length=500,
        description="用户输入的问题或卡牌描述",
        examples=["龙之战士有什么效果？"],
    )


class QueryRequest(BaseModel):
    """RAG 查询请求"""
    user_input: str = Field(
        ...,
        max_length=500,
        description="用户输入的问题",
        examples=["龙之战士入场效果怎么触发？"],
    )
    selected_cardnos: list[str] = Field(
        default=[],
        description="用户选择的卡牌编号列表，空列表表示使用自动识别",
        examples=[["BP01-001"]],
    )
    enable_llm: bool = Field(
        default=True,
        description="是否启用 LLM 生成回答。false 时仅返回向量检索结果",
    )
    use_lora: bool = Field(
        default=False,
        description="是否使用 LoRA 微调后的 Embedding 模型",
    )


# ============================================================
# 响应模型
# ============================================================

class CardCandidate(BaseModel):
    """候选卡牌"""
    cardno: str
    name: str
    description: str = ""
    card_type: str = ""
    class_: str = Field(default="", alias="class")
    cost: str = ""
    power: str = ""
    hp: str = ""
    score: float = 0.0


class IdentifyResponse(BaseModel):
    """卡牌识别响应"""
    candidates: list[CardCandidate]


class QAResult(BaseModel):
    """关联 QA 结果"""
    cardno: str
    name: str
    qa: dict


class SearchResultItem(BaseModel):
    """向量检索结果项"""
    id: str = ""
    text: str = ""
    score: float = 0.0
    source: str = ""
    cardno: str = ""
    card_name: str = ""


class QueryResponse(BaseModel):
    """RAG 查询响应"""
    answer: Optional[str] = Field(
        default=None,
        description="LLM 生成的回答，LLM 关闭时为 null",
    )
    matched_cards: list[CardCandidate] = Field(
        default_factory=list,
        description="识别的卡牌",
    )
    qa_results: list[QAResult] = Field(
        default_factory=list,
        description="关联的 QA",
    )
    search_results: list[SearchResultItem] = Field(
        default_factory=list,
        description="向量检索结果",
    )
    context: str = Field(
        default="",
        description="构建的 LLM 上下文（调试用）",
    )
    prompt: str = Field(
        default="",
        description="完整 LLM Prompt（调试用）",
    )


class FaissIndicesStatus(BaseModel):
    """FAISS 索引状态"""
    card_names: bool = False
    content: bool = False
    content_lora: bool = False


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    embedder_loaded: bool = False
    faiss_indices: FaissIndicesStatus = FaissIndicesStatus()
    llm_enabled: bool = False
    llm_provider: str = ""
    llm_reachable: bool = False
