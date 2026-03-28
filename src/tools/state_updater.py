"""
游戏状态修改工具 — 通过 Function Calling 实现"言出法随"。

这些工具函数由 LLM 决定是否调用，
执行后直接修改 GameState 中的数值。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.models.game_state import GameState
from src.tools.base import register_tool


@register_tool(
    name="update_affection",
    description="修改某个角色对另一个角色的好感度。正值增加好感，负值降低好感。",
    parameters={
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "好感度的主体（谁对谁有好感），通常是 NPC 的 ID",
            },
            "target_id": {
                "type": "string",
                "description": "好感度的客体，通常是 'player'",
            },
            "delta": {
                "type": "number",
                "description": "好感度变化值，范围 -20 到 +20",
            },
            "reason": {
                "type": "string",
                "description": "变化原因（用于日志记录）",
            },
        },
        "required": ["source_id", "target_id", "delta"],
    },
)
def update_affection(
    game_state: GameState,
    source_id: str,
    target_id: str,
    delta: float,
    reason: str = "",
    **kwargs,
) -> dict[str, Any]:
    """修改好感度。"""
    new_value = game_state.update_affection(source_id, target_id, delta)
    logger.info(
        f"[Tool:update_affection] {source_id} -> {target_id}: "
        f"好感度 {'+' if delta >= 0 else ''}{delta} = {new_value} ({reason})"
    )
    return {
        "source_id": source_id,
        "target_id": target_id,
        "new_affection": new_value,
        "delta": delta,
        "reason": reason,
    }


@register_tool(
    name="update_player_location",
    description="修改玩家的当前位置。",
    parameters={
        "type": "object",
        "properties": {
            "new_location": {
                "type": "string",
                "description": "新的地点名称",
            },
        },
        "required": ["new_location"],
    },
)
def update_player_location(
    game_state: GameState,
    new_location: str,
    **kwargs,
) -> dict[str, Any]:
    """修改玩家位置。"""
    old = game_state.player.location
    game_state.player.location = new_location
    logger.info(f"[Tool:update_location] 玩家位置: {old} -> {new_location}")
    return {"old_location": old, "new_location": new_location}


@register_tool(
    name="set_world_flag",
    description="设置一个全局事件标记，表示游戏世界中发生了某个重要事件。",
    parameters={
        "type": "object",
        "properties": {
            "flag_name": {
                "type": "string",
                "description": "标记名称，如 'vase_broken'、'secret_revealed'",
            },
            "flag_value": {
                "type": "string",
                "description": "标记值",
            },
        },
        "required": ["flag_name", "flag_value"],
    },
)
def set_world_flag(
    game_state: GameState,
    flag_name: str,
    flag_value: str,
    **kwargs,
) -> dict[str, Any]:
    """设置全局事件标记。"""
    game_state.world_flags[flag_name] = flag_value
    logger.info(f"[Tool:set_world_flag] {flag_name} = {flag_value}")
    return {"flag_name": flag_name, "flag_value": flag_value}
