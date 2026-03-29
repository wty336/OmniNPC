from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.memory import MemoryQueryResult


class MemorySnapshot(MemoryQueryResult):
    """结构化记忆快照，作为记忆适配器的统一输出。"""


class MemoryAdapter(ABC):
    @abstractmethod
    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        raise NotImplementedError

    @abstractmethod
    def consolidate(self, **kwargs) -> None:
        raise NotImplementedError
