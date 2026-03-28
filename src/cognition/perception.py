"""
感知模块 (Perception) — 认知管线的第一步。

职责：
1. 接收玩家输入
2. 触发三层记忆检索
3. 组装完整的上下文（Context）供后续模块使用
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.models.character import CharacterProfile
from src.models.game_state import GameState
from src.models.memory import MemoryQueryResult
from src.memory.memory_manager import MemoryManager


class PerceptionContext(BaseModel):
    """感知模块输出的上下文对象，供后续认知步骤使用。"""

    player_input: str
    character: CharacterProfile
    game_state: GameState
    memory_result: MemoryQueryResult
    environment_desc: str = Field(default="", description="当前环境描述")

    class Config:
        arbitrary_types_allowed = True


class Perception:
    """
    感知模块。

    将玩家输入 + 记忆检索结果 + 当前游戏状态
    打包为结构化的 PerceptionContext。
    """

    def __init__(self, memory_manager: MemoryManager):
        self._memory = memory_manager

    def perceive(
        self,
        player_input: str,
        character: CharacterProfile,
        game_state: GameState,
    ) -> PerceptionContext:
        """
        执行感知：检索记忆 + 组装上下文。

        Parameters
        ----------
        player_input : 玩家输入文本
        character : NPC 角色档案
        game_state : 当前游戏状态

        Returns
        -------
        PerceptionContext : 完整的感知上下文
        """
        logger.info(f"[Perception] 开始感知: input='{player_input[:50]}...'")

        # 1. 检索三层记忆
        memory_result = self._memory.retrieve(
            query=player_input,
            player_id=game_state.player.player_id,
        )

        # 2. 构建环境描述
        rel = game_state.get_relationship(
            game_state.player.player_id, character.id
        )
        environment_desc = (
            f"当前地点：{game_state.player.location}\n"
            f"玩家「{game_state.player.name}」对「{character.name}」的好感度：{rel.affection}\n"
            f"玩家持有道具：{', '.join(game_state.player.inventory) or '无'}"
        )

        context = PerceptionContext(
            player_input=player_input,
            character=character,
            game_state=game_state,
            memory_result=memory_result,
            environment_desc=environment_desc,
        )

        logger.debug(
            f"[Perception] 感知完成: memories(working={len(memory_result.working_memories)}, "
            f"episodic={len(memory_result.episodic_memories)}, "
            f"facts={len(memory_result.semantic_facts)})"
        )
        return context
