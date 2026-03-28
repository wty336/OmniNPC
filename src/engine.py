"""
OmniNPC 引擎核心 — NPC 会话管理与认知管线调度。

管理多个 NPC 的角色加载、记忆实例化和认知管线调度，
对外提供统一的 process_chat() 接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from src.cognition.pipeline import CognitivePipeline
from src.memory.memory_manager import MemoryManager
from src.models.character import CharacterProfile
from src.models.game_state import GameState
from src.models.message import AgentResponse
from src.storage.game_state_store import GameStateStore

# 确保工具已注册（导入时触发装饰器）
import src.tools.state_updater  # noqa: F401
import src.tools.item_manager   # noqa: F401


class NPCEngine:
    """
    OmniNPC 引擎核心。

    管理:
    - 角色加载与缓存
    - 每个 NPC 的记忆管理器与认知管线
    - 游戏状态持久化
    """

    def __init__(self):
        self._characters: dict[str, CharacterProfile] = {}
        self._memory_managers: dict[str, MemoryManager] = {}
        self._pipelines: dict[str, CognitivePipeline] = {}
        self._state_store = GameStateStore()
        
        from src.sandbox.tick_engine import TickEngine
        from src.sandbox.rumor_spreader import RumorSpreader
        self.tick_engine = TickEngine()
        self.rumor_spreader = RumorSpreader()
        
        logger.info("[NPCEngine] 引擎初始化")

    def load_character(self, character_id: str, yaml_path: Optional[str] = None) -> CharacterProfile:
        """
        加载或获取 NPC 角色。

        Parameters
        ----------
        character_id : 角色 ID
        yaml_path : YAML 配置文件路径（不提供则自动查找）
        """
        if character_id in self._characters:
            return self._characters[character_id]

        if yaml_path is None:
            yaml_path = str(
                Path(settings.characters_dir) / f"{character_id}.yaml"
            )

        character = CharacterProfile.from_yaml(yaml_path)
        self._characters[character_id] = character

        # 初始化该 NPC 的记忆管理器
        memory = MemoryManager(character_id)
        # 从角色配置初始化关系图谱
        memory.init_relationships(
            [rel.model_dump() for rel in character.initial_relationships]
        )
        self._memory_managers[character_id] = memory

        # 初始化认知管线
        self._pipelines[character_id] = CognitivePipeline(memory)

        logger.info(f"[NPCEngine] 加载角色: {character.name} ({character_id})")
        return character

    def process_chat(
        self,
        player_input: str,
        character_id: str,
        session_id: str = "default_session",
    ) -> AgentResponse:
        """
        处理一次对话交互。

        Parameters
        ----------
        player_input : 玩家输入文本
        character_id : 对话目标 NPC ID
        session_id : 会话 ID

        Returns
        -------
        AgentResponse : 完整的引擎响应
        """
        # 确保角色已加载
        if character_id not in self._characters:
            self.load_character(character_id)

        character = self._characters[character_id]
        pipeline = self._pipelines[character_id]

        # 加载游戏状态
        game_state = self._state_store.load_or_create(session_id)

        # 自动更新玩家位置为当前对话 NPC 的所在地
        if character.location:
            game_state.player.location = character.location

        # 同步 NPC 运行时状态
        from src.models.game_state import NPCState
        if character_id not in game_state.npcs:
            game_state.npcs[character_id] = NPCState(
                character_id=character_id,
                location=character.location,
            )

        # 执行认知管线
        response = pipeline.run(
            player_input=player_input,
            character=character,
            game_state=game_state,
        )

        # 持久化游戏状态
        self._state_store.save(game_state)

        return response

    def get_game_state(self, session_id: str) -> GameState:
        """获取游戏状态。"""
        return self._state_store.load_or_create(session_id)

    def force_tick(self) -> dict:
        """手动触发一次全局 Tick。"""
        # 1. 触发 TickEngine（处理定时操作）
        tick_res = self.tick_engine.tick(engine=self)
        
        # 2. 触发流言传播
        def _get_mm(cid):
            if cid not in self._memory_managers:
                self.load_character(cid)
            return self._memory_managers.get(cid)
            
        # 让尚未加载的硬编码角色也参与传播（用于测试）
        all_ids = list(set(["tsundere_sister", "gentle_healer"] + list(self._characters.keys())))
        
        spreads = self.rumor_spreader.spread_tick(
            all_npc_ids=all_ids,
            get_memory_manager=_get_mm,
        )
        tick_res["rumor_spreads"] = spreads
        return tick_res

    @property
    def loaded_characters(self) -> list[str]:
        """获取已加载的角色 ID 列表。"""
        return list(self._characters.keys())


# 全局引擎单例
_engine: Optional[NPCEngine] = None


def get_engine() -> NPCEngine:
    """获取引擎全局单例。"""
    global _engine
    if _engine is None:
        _engine = NPCEngine()
    return _engine
