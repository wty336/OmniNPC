from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

from src.adapters.memory.base import MemoryAdapter, MemorySnapshot
from src.adapters.memory.composite import CompositeMemoryAdapter
from src.models.memory import ConversationTurn, MemoryItem, MemoryQueryResult, MemoryType


class FakeMemoryAdapter(MemoryAdapter):
    def __init__(self) -> None:
        self.last_consolidated = None

    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        return MemorySnapshot(
            working_memories=[
                ConversationTurn(role="player", speaker_name="玩家", content=query)
            ],
            episodic_memories=[
                MemoryItem(
                    memory_type=MemoryType.EPISODIC,
                    content="玩家与 npc 的关系为「友人」(权重: 0.6)",
                    summary="关系摘要",
                )
            ],
            semantic_facts=["player 与 npc 的关系为「友人」(权重: 0.6)"],
            graph_relations=[],
        )

    def consolidate(self, **kwargs) -> None:
        self.last_consolidated = kwargs


class FakeMemoryManager:
    def __init__(self) -> None:
        self.retrievals: list[tuple[str, str]] = []
        self.consolidations: list[dict[str, object]] = []

    def retrieve(self, query: str, player_id: str = "player_default") -> MemoryQueryResult:
        self.retrievals.append((query, player_id))
        return MemoryQueryResult(
            working_memories=[
                ConversationTurn(role="player", speaker_name="玩家", content=query)
            ],
            episodic_memories=[],
            semantic_facts=["来自管理器的事实"],
            graph_relations=[],
        )

    def consolidate(self, **kwargs) -> None:
        self.consolidations.append(kwargs)


def test_memory_adapter_contract_and_composite_delegation():
    adapter = FakeMemoryAdapter()

    snapshot = adapter.retrieve("你好")
    assert isinstance(snapshot, MemorySnapshot)
    assert "友人" in snapshot.semantic_facts[0]

    manager = FakeMemoryManager()
    composite = CompositeMemoryAdapter(manager)

    result = composite.retrieve("你好", player_id="player-1")
    assert isinstance(result, MemorySnapshot)
    assert result.semantic_facts == ["来自管理器的事实"]
    assert manager.retrievals == [("你好", "player-1")]

    composite.consolidate(player_input="你好", npc_response="嗯", emotion_score=7.0)
    assert manager.consolidations == [
        {"player_input": "你好", "npc_response": "嗯", "emotion_score": 7.0}
    ]
