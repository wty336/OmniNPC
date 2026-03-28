"""
FastAPI 应用入口 — 中间件、生命周期、CORS 配置。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info(f"🚀 {settings.app_name} 引擎启动中...")
    # 启动时预加载（可选）
    yield
    logger.info(f"🛑 {settings.app_name} 引擎已关闭")


app = FastAPI(
    title=settings.app_name,
    description="面向高自由度互动游戏（AVG）的认知架构 Agent 引擎",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from src.api.routes.chat import router as chat_router  # noqa: E402

app.include_router(chat_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """健康检查。"""
    return {"status": "ok", "app": settings.app_name}
