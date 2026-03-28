"""
道具管理工具 — 增删玩家道具。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.models.game_state import GameState
from src.tools.base import register_tool


@register_tool(
    name="add_item",
    description="给玩家添加一个道具。",
    parameters={
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": "道具名称",
            },
        },
        "required": ["item_name"],
    },
)
def add_item(
    game_state: GameState, item_name: str, **kwargs
) -> dict[str, Any]:
    """添加道具。"""
    if item_name not in game_state.player.inventory:
        game_state.player.inventory.append(item_name)
    logger.info(f"[Tool:add_item] 获得道具: {item_name}")
    return {"item_name": item_name, "action": "added"}


@register_tool(
    name="remove_item",
    description="从玩家背包中移除一个道具。",
    parameters={
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": "道具名称",
            },
        },
        "required": ["item_name"],
    },
)
def remove_item(
    game_state: GameState, item_name: str, **kwargs
) -> dict[str, Any]:
    """移除道具。"""
    if item_name in game_state.player.inventory:
        game_state.player.inventory.remove(item_name)
        logger.info(f"[Tool:remove_item] 移除道具: {item_name}")
        return {"item_name": item_name, "action": "removed"}
    else:
        logger.warning(f"[Tool:remove_item] 道具不存在: {item_name}")
        return {"item_name": item_name, "action": "not_found"}
