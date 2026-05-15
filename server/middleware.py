"""
FastAPI 中间件 —— 全局速率限制、请求日志。

所有中间件都是无用户维度的全局控制。
输入验证（长度/Prompt Injection）已移至 FastAPI 依赖注入，避免消费请求体。
"""

import logging
import re
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Injection 检测模式
# ============================================================
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(your\s+)?(previous\s+)?prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(\w+\s+){0,3}(bot|assistant|dan|jailbreak)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are", re.IGNORECASE),
]


def detect_injection(text: str) -> bool:
    """检测输入中是否包含 Prompt Injection 模式"""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ============================================================
# FastAPI 依赖：输入验证（在路由层执行，不消费请求体）
# ============================================================

class InputValidator:
    """
    输入验证依赖 —— 配合 Pydantic 模型使用。
    Pydantic 已处理长度验证（max_length=500），此处仅做注入检测。
    使用方式：在路由函数参数中添加 `_validator: None = Depends(InputValidator(...))`
    """

    def __init__(self, user_input: str):
        if detect_injection(user_input):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="输入包含不被允许的内容",
            )


# ============================================================
# 全局速率限制中间件
# ============================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    全局速率限制（基于内存，无用户维度）：
    - 对所有 /api/ 路径生效
    - LLM 路径有独立的更严格限制
    - 使用滑动窗口计数
    """

    _LLM_PATHS = {"/api/query"}

    def __init__(self, app):
        super().__init__(app)
        self._global_requests: dict[int, list[float]] = defaultdict(list)
        self._llm_requests: dict[int, list[float]] = defaultdict(list)

    def _check_limit(
        self,
        history: list[float],
        max_count: int,
        window_seconds: int = 60,
    ) -> bool:
        """检查时间窗口内是否超过限制"""
        now = time.time()
        cutoff = now - window_seconds
        history[:] = [t for t in history if t > cutoff]
        if len(history) >= max_count:
            return False
        history.append(now)
        return True

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        if not path.startswith("/api/"):
            return await call_next(request)

        # 跳过健康检查
        if path == "/api/health":
            return await call_next(request)

        # 全局限制
        if not self._check_limit(
            self._global_requests[0],
            settings.global_rate_limit_per_minute,
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="请求过于频繁，请稍后再试",
                headers={"Retry-After": "60"},
            )

        # LLM 路径额外限制（仅全局开启 LLM 时生效）
        if path in self._LLM_PATHS and settings.llm_enabled:
            if not self._check_limit(
                self._llm_requests[0],
                settings.llm_rate_limit_per_minute,
            ):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="LLM 调用过于频繁，请稍后再试或关闭 LLM 开关",
                    headers={"Retry-After": "60"},
                )

        return await call_next(request)


# ============================================================
# 请求日志中间件
# ============================================================
class RequestLogMiddleware(BaseHTTPMiddleware):
    """记录所有 API 请求的方法、路径、耗时、状态码"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        elapsed = time.time() - start_time

        logger.info(
            "%s %s → %d (%.3fs)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
