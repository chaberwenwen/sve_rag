"""
FastAPI 应用入口 —— 挂载中间件、路由，启动 uvicorn。

启动方式:
    python -m server.main
    或
    uvicorn server.main:app --host 0.0.0.0 --port 8001
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from server.middleware import (
    RateLimitMiddleware,
    RequestLogMiddleware,
)
from server.api.identify import router as identify_router
from server.api.query import router as query_router
from server.api.health import router as health_router

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="SVE RAG 查询助手",
    description="影之诗 EVOLVE 卡牌规则 RAG 检索系统 - 公开 Web API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ============================================================
# CORS（允许前端跨域访问）
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 公开服务，允许所有来源
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ============================================================
# 中间件（按顺序添加，先添加的后执行）
# ============================================================
app.add_middleware(RequestLogMiddleware)
app.add_middleware(RateLimitMiddleware)

# ============================================================
# 路由注册
# ============================================================
app.include_router(identify_router)
app.include_router(query_router)
app.include_router(health_router)


# ============================================================
# 前端页面（根路径返回 index.html）
# ============================================================
_web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
_index_html = os.path.join(_web_dir, "index.html")


@app.get("/")
async def root():
    """返回前端 SPA 页面"""
    if os.path.isfile(_index_html):
        return FileResponse(_index_html, media_type="text/html; charset=utf-8")
    return {"message": "SVE RAG API is running", "docs": "/api/docs"}


# ============================================================
# 启动事件
# ============================================================
@app.on_event("startup")
async def startup_event():
    """服务启动时预热加载"""
    logger.info("🚀 SVE RAG 服务启动中...")
    # 预热 Embedder（后台加载，不阻塞）
    try:
        from server.model_pool import get_embedder
        embedder = get_embedder()
        embedder._ensure_loaded()
        logger.info("✅ Embedder 加载完成")
    except Exception as e:
        logger.warning("⚠️ Embedder 预热失败（将在首次请求时懒加载）: %s", e)


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    logger.info("正在关闭服务...")
    try:
        from server.llm_async import get_llm_client
        client = get_llm_client()
        await client.close()
        logger.info("✅ LLM 客户端已关闭")
    except Exception:
        pass


# ============================================================
# 直接启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
