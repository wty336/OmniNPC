"""消息与响应数据模型 — API 通信与引擎内部传递的标准格式。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# API 请求 / 响应
# ============================================================

class ChatRequest(BaseModel):
    """前端发给 API 的对话请求。"""

    player_input: str = Field(description="玩家输入的文本")
    character_id: str = Field(description="对话目标 NPC 的 ID")
    session_id: str = Field(default="default_session")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="可选的额外上下文（如当前地点、时间等）",
    )


class ToolCallResult(BaseModel):
    """单次工具调用的结果。"""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """引擎返回给前端的完整响应。"""

    # 玩家可见内容
    dialogue: str = Field(description="NPC 台词（玩家可见）")
    emotion: str = Field(default="neutral", description="NPC 当前表情/情绪标签")

    # 调试信息（可选暴露）
    inner_monologue: Optional[str] = Field(
        default=None,
        description="NPC 内心独白（debug 模式下返回）",
    )

    # 状态变更
    tool_calls: list[ToolCallResult] = Field(
        default_factory=list,
        description="本次触发的工具调用及结果",
    )
    state_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="状态变更摘要（前端可据此触发 UI 更新）",
    )

    # 元信息
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="附加元数据（情绪打分、实体抽取结果等）",
    )
    character_id: str = ""
    character_name: str = ""


class ChatResponse(BaseModel):
    """API 层最终返回给前端的标准响应。"""

    success: bool = True
    data: Optional[AgentResponse] = None
    error: Optional[str] = None


# ============================================================
# LLM 内部消息格式
# ============================================================

class LLMMessage(BaseModel):
    """与 LLM 通信的标准消息格式。"""

    role: str = Field(description="system / user / assistant / tool")
    content: str = Field(default="")
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
