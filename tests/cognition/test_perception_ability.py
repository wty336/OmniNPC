from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

from src.adapters.memory.base import MemoryAdapter, MemorySnapshot
from src.cognition.perception import Perception
from src.models.character import CharacterProfile, Personality
from src.models.game_state import GameState
from src.models.memory import ConversationTurn, MemoryItem, MemoryType


class FakeMemoryAdapter(MemoryAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        self.calls.append((query, player_id))
        return MemorySnapshot(
            working_memories=[
                ConversationTurn(role="player", speaker_name="旅人", content=query)
            ],
            episodic_memories=[
                MemoryItem(
                    memory_type=MemoryType.EPISODIC,
                    content="曾经收到过玩家的帮助",
                    summary="旧记忆",
                )
            ],
            semantic_facts=["player 与 npc 的关系为「友人」"],
            graph_relations=[],
        )

    def consolidate(self, **kwargs) -> None:
        raise AssertionError("perception should not consolidate memory")


class FakeMemoryManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        self.calls.append((query, player_id))
        return MemorySnapshot(
            working_memories=[],
            episodic_memories=[],
            semantic_facts=["来自旧管理器的事实"],
            graph_relations=[],
        )


def make_character() -> CharacterProfile:
    return CharacterProfile(
        id="npc_1",
        name="凌霜",
        role="师姐",
        personality=Personality(traits=["冷淡"], speaking_style="简短"),
        system_prompt="你是凌霜。",
    )


def make_game_state() -> GameState:
    game_state = GameState(session_id="session-1")
    game_state.player.player_id = "player-42"
    game_state.player.name = "行者"
    game_state.player.location = "山门"
    game_state.player.inventory = ["玉佩"]
    return game_state


def test_perception_reads_from_injected_memory_adapter():
    adapter = FakeMemoryAdapter()
    perception = Perception(memory_adapter=adapter)

    context = perception.perceive("你好", make_character(), make_game_state())

    assert adapter.calls == [("你好", "player-42")]
    assert context.memory_result.semantic_facts == ["player 与 npc 的关系为「友人」"]
    assert "山门" in context.environment_desc
    assert "玉佩" in context.environment_desc


def test_perception_accepts_legacy_memory_manager_argument():
    memory_manager = FakeMemoryManager()
    perception = Perception(memory_manager)

    context = perception.perceive("救我", make_character(), make_game_state())

    assert memory_manager.calls == [("救我", "player-42")]
    assert context.memory_result.semantic_facts == ["来自旧管理器的事实"]
