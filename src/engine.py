"""
OmniNPC 引擎核心 — NPC 会话管理与认知管线调度。

管理多个 NPC 的角色加载、记忆实例化和认知管线调度，
对外提供统一的 process_chat() 接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from config.settings import settings
from src.memory.memory_manager import MemoryManager
from src.models.character import CharacterProfile
from src.models.game_state import GameState
from src.models.message import AgentResponse
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.runtime_result import RuntimeResult
from src.runtime.turn_context import TurnContext
from src.storage.game_state_store import GameStateStore
# 确保工具已注册（导入时触发装饰器）
import src.tools.state_updater  # noqa: F401
import src.tools.item_manager   # noqa: F401

if TYPE_CHECKING:
    from src.cognition.pipeline import CognitivePipeline


class NPCEngine:
    """
    OmniNPC 引擎核心。

    管理:
    - 角色加载与缓存
    - 每个 NPC 的记忆管理器与认知管线
    - 游戏状态持久化
    """

    def __init__(self, use_agent_runtime: bool = False):
        self._characters: dict[str, CharacterProfile] = {}
        self._memory_managers: dict[str, MemoryManager] = {}
        self._pipelines: dict[str, CognitivePipeline] = {}
        self._runtimes: dict[str, AgentRuntime] = {}
        self._use_agent_runtime = use_agent_runtime
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

        logger.info(f"[NPCEngine] 加载角色: {character.name} ({character_id})")
        return character

    def _get_or_create_pipeline(self, character_id: str) -> CognitivePipeline:
        pipeline = self._pipelines.get(character_id)
        if pipeline is None:
            from src.cognition.pipeline import CognitivePipeline

            pipeline = CognitivePipeline(self._memory_managers[character_id])
            self._pipelines[character_id] = pipeline
        return pipeline

    def _get_or_create_runtime(self, character_id: str) -> AgentRuntime:
        runtime = self._runtimes.get(character_id)
        if runtime is None:
            from src.adapters.memory.composite import CompositeMemoryAdapter
            from src.adapters.tools.catalog import ToolCatalog
            from src.adapters.tools.executor import ToolExecutor
            from src.cognition.action_planner import ActionPlanner
            from src.cognition.inner_monologue import InnerMonologue
            from src.cognition.perception import Perception
            from src.runtime.chat_policy import ChatRuntimePolicy

            character = self._characters[character_id]
            memory_adapter = CompositeMemoryAdapter(
                self._memory_managers[character_id]
            )
            runtime = AgentRuntime(
                policy=ChatRuntimePolicy(),
                tool_executor=ToolExecutor(ToolCatalog.load()),
                memory_adapter=memory_adapter,
                reflector=InnerMonologue(),
                action_planner=ActionPlanner(),
                perception=Perception(memory_adapter=memory_adapter),
                character=character,
            )
            self._runtimes[character_id] = runtime
        return runtime

    def _adapt_runtime_result(
        self,
        character: CharacterProfile,
        result: RuntimeResult,
    ) -> AgentResponse:
        """将 runtime 输出适配为现有 API 使用的 AgentResponse。"""
        return AgentResponse(
            dialogue=result.dialogue,
            emotion="neutral",
            inner_monologue=None,
            tool_calls=[],
            state_changes={},
            metadata=None,
            character_id=character.id,
            character_name=character.name,
        )

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

        if self._use_agent_runtime:
            # Runtime is the default chat path; keep legacy behind explicit fallback only.
            turn_context = TurnContext(
                turn_id=f"{session_id}:{character_id}",
                session_id=session_id,
                character_id=character_id,
                player_input=player_input,
                max_steps=6,
            )
            runtime = self._get_or_create_runtime(character_id)
            runtime_result = runtime.run(turn_context, game_state=game_state)
            response = self._adapt_runtime_result(character, runtime_result)
        else:
            # Legacy compatibility path. Keep available until runtime behavior fully replaces it.
            pipeline = self._get_or_create_pipeline(character_id)

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

    def get_memory_manager(self, character_id: str) -> MemoryManager | None:
        """获取或延迟加载指定角色的记忆管理器。"""
        if character_id not in self._memory_managers and character_id not in self._characters:
            self.load_character(character_id)
        return self._memory_managers.get(character_id)

    def iter_memory_managers(self):
        """公开遍历已加载角色的记忆管理器，避免外部直接依赖私有字段。"""
        return self._memory_managers.items()

    def force_tick(self) -> dict:
        """手动触发一次全局 Tick。"""
        # 1. 触发 TickEngine（处理定时操作）
        tick_res = self.tick_engine.tick(engine=self)
        
        # 2. 触发流言传播
        # 让尚未加载的硬编码角色也参与传播（用于测试）
        all_ids = list(set(["tsundere_sister", "gentle_healer"] + list(self._characters.keys())))
        
        spreads = self.rumor_spreader.spread_tick(
            all_npc_ids=all_ids,
            get_memory_manager=self.get_memory_manager,
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
        _engine = NPCEngine(use_agent_runtime=True)
    return _engine
