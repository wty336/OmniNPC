"""
对话路由 — POST /api/v1/chat
"""

from fastapi import APIRouter, HTTPException

from loguru import logger

from src.engine import get_engine
from src.models.message import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与 NPC 对话。

    请求示例:
    ```json
    {
        "player_input": "师姐，我把花瓶打碎了...",
        "character_id": "tsundere_sister",
        "session_id": "my_session"
    }
    ```
    """
    try:
        engine = get_engine()
        response = await engine.process_chat(
            player_input=request.player_input,
            character_id=request.character_id,
            session_id=request.session_id,
        )
        return ChatResponse(success=True, data=response)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"角色配置未找到: {request.character_id}",
        )
    except Exception as e:
        logger.error(f"对话处理失败: {e}", exc_info=True)
        return ChatResponse(success=False, error=str(e))


@router.get("/characters")
async def list_characters():
    """列出已加载的 NPC 角色。"""
    engine = get_engine()
    return {"characters": engine.loaded_characters}


@router.get("/state/{session_id}")
async def get_state(session_id: str):
    """获取游戏状态。"""
    engine = get_engine()
    state = engine.get_game_state(session_id)
    return state.model_dump()
