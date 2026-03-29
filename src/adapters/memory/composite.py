from __future__ import annotations

from typing import Any

from src.adapters.memory.base import MemoryAdapter, MemorySnapshot


class CompositeMemoryAdapter(MemoryAdapter):
    def __init__(self, memory_manager: Any):
        self._memory_manager = memory_manager

    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        result = self._memory_manager.retrieve(query=query, player_id=player_id)
        return MemorySnapshot.model_validate(result.model_dump())

    def consolidate(self, **kwargs) -> None:
        self._memory_manager.consolidate(**kwargs)
