"""
工作记忆 (Working Memory) — 短期滑动窗口。

保留最近 N 轮对话，作为 LLM 上下文的一部分直接注入 prompt。
这是最"即时"的记忆层，类比人类的工作记忆 / 注意力缓存。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from loguru import logger

from config.settings import settings
from src.models.memory import ConversationTurn


class WorkingMemory:
    """
    基于滑动窗口的工作记忆。

    每个 NPC 拥有独立的工作记忆实例。
    窗口大小由 settings.memory.working_memory_window 控制。
    """

    def __init__(self, character_id: str, window_size: Optional[int] = None):
        self.character_id = character_id
        self._window_size = window_size or settings.memory.working_memory_window
        self._buffer: deque[ConversationTurn] = deque(maxlen=self._window_size)
        logger.debug(
            f"[WorkingMemory] 初始化: character={character_id}, window={self._window_size}"
        )

    def add(self, turn: ConversationTurn) -> None:
        """添加一轮对话到工作记忆。超出窗口的旧记录自动淘汰。"""
        self._buffer.append(turn)

    def get_recent(self, n: Optional[int] = None) -> list[ConversationTurn]:
        """获取最近 n 轮对话（默认返回全部窗口内容）。"""
        items = list(self._buffer)
        if n is not None:
            return items[-n:]
        return items

    def to_messages(self) -> list[dict[str, str]]:
        """
        将工作记忆转换为 LLM 消息格式。

        Returns
        -------
        list[dict] : [{"role": "user/assistant", "content": "..."}]
        """
        messages = []
        for turn in self._buffer:
            if turn.role == "player":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "npc":
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.role == "system":
                messages.append({"role": "system", "content": turn.content})
        return messages

    def clear(self) -> None:
        """清空工作记忆。"""
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
