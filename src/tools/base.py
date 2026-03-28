"""
工具系统基座 — 注册器 + Function Calling 定义生成。

通过 @register_tool 装饰器注册工具函数，
自动生成 OpenAI Function Calling 格式的工具定义列表。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from loguru import logger

# 全局工具注册表
_TOOL_REGISTRY: dict[str, Callable] = {}
_TOOL_DEFINITIONS: list[dict[str, Any]] = []


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
):
    """
    工具注册装饰器。

    Usage
    -----
    @register_tool(
        name="update_affection",
        description="修改玩家与 NPC 之间的好感度",
        parameters={
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "description": "NPC ID"},
                "delta": {"type": "number", "description": "好感度变化值"}
            },
            "required": ["target_id", "delta"]
        }
    )
    async def update_affection(game_state, target_id, delta):
        ...
    """

    def decorator(func: Callable) -> Callable:
        _TOOL_REGISTRY[name] = func
        _TOOL_DEFINITIONS.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
        logger.debug(f"[ToolRegistry] 注册工具: {name}")
        return func

    return decorator


def get_tool_registry() -> dict[str, Callable]:
    """获取工具注册表。"""
    return _TOOL_REGISTRY


def get_tool_definitions() -> list[dict[str, Any]]:
    """获取 OpenAI Function Calling 格式的工具定义列表。"""
    return _TOOL_DEFINITIONS
