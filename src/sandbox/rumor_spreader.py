"""
流言传播模块 (Rumor Spreader) — NPC 间异步信息扩散。

模拟信息在 NPC 社交网络中的传播：
1. NPC A 对玩家说了一件事
2. NPC A 在离线 Tick 中可能将这件事告诉 NPC B
3. NPC B 下次见到玩家时可能提到这件事（但可能有偏差）

这是实现"世界活起来"感觉的核心机制之一。
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from loguru import logger

from src.models.memory import MemoryItem, MemoryType


class Rumor:
    """一条流言。"""

    def __init__(
        self,
        content: str,
        source_npc: str,
        original_event: str,
        credibility: float = 1.0,
        spread_count: int = 0,
    ):
        self.content = content
        self.source_npc = source_npc
        self.original_event = original_event
        self.credibility = credibility  # 可信度，每次传播递减
        self.spread_count = spread_count
        self.created_at = time.time()
        self.heard_by: set[str] = {source_npc}

    def degrade(self, factor: float = 0.85) -> "Rumor":
        """流言传播后可信度降低。"""
        self.credibility *= factor
        self.spread_count += 1
        return self


class RumorSpreader:
    """
    流言传播器。

    管理 NPC 间的信息扩散，
    支持在离线 Tick 中自动传播。
    """

    def __init__(self):
        self._active_rumors: list[Rumor] = []
        self._spread_probability: float = 0.3  # 每次 Tick 的传播概率

    def create_rumor(
        self,
        content: str,
        source_npc: str,
        original_event: str,
    ) -> Rumor:
        """
        创建一条新流言。

        Parameters
        ----------
        content : 流言内容
        source_npc : 传播源 NPC ID
        original_event : 原始事件描述
        """
        rumor = Rumor(
            content=content,
            source_npc=source_npc,
            original_event=original_event,
        )
        self._active_rumors.append(rumor)
        logger.info(
            f"[RumorSpreader] 新流言: '{content[:50]}...' "
            f"来源={source_npc}"
        )
        return rumor

    def spread_tick(
        self,
        all_npc_ids: list[str],
        get_memory_manager=None,
    ) -> list[dict[str, Any]]:
        """
        执行一次流言传播 Tick。

        对每条活跃流言，以一定概率传播给尚未听到的 NPC。

        Parameters
        ----------
        all_npc_ids : 所有已加载的 NPC ID 列表
        get_memory_manager : 获取 MemoryManager 的回调函数

        Returns
        -------
        list[dict] : 传播记录 [{"rumor": ..., "from": ..., "to": ..., "credibility": ...}]
        """
        spread_records = []

        for rumor in self._active_rumors[:]:
            # 跳过可信度过低的流言
            if rumor.credibility < 0.1:
                self._active_rumors.remove(rumor)
                continue

            # 找出还没听到这条流言的 NPC
            potential_targets = [
                npc_id for npc_id in all_npc_ids
                if npc_id not in rumor.heard_by
            ]

            if not potential_targets:
                continue

            # 按概率决定是否传播
            if random.random() > self._spread_probability:
                continue

            # 随机选一个目标 NPC
            target_npc = random.choice(potential_targets)
            rumor.heard_by.add(target_npc)
            rumor.degrade()

            # 将流言写入目标 NPC 的情景记忆
            if get_memory_manager:
                try:
                    mm = get_memory_manager(target_npc)
                    if mm:
                        memory = MemoryItem(
                            memory_type=MemoryType.EPISODIC,
                            content=f"[听闻] {rumor.content}",
                            summary=f"听{rumor.source_npc}说的传闻",
                            emotion_score=3.0,  # 听闻的流言情绪影响较低
                            importance=4.0,
                            character_id=target_npc,
                            related_entities=[rumor.source_npc],
                            metadata={
                                "type": "rumor",
                                "credibility": rumor.credibility,
                                "original_source": rumor.source_npc,
                            },
                        )
                        mm.episodic.store(memory)
                except Exception as e:
                    logger.warning(f"[RumorSpreader] 写入记忆失败: {e}")

            record = {
                "rumor_content": rumor.content[:80],
                "from": rumor.source_npc,
                "to": target_npc,
                "credibility": rumor.credibility,
                "spread_count": rumor.spread_count,
            }
            spread_records.append(record)

            logger.info(
                f"[RumorSpreader] 📢 流言传播: "
                f"{rumor.source_npc} → {target_npc} "
                f"(可信度={rumor.credibility:.2f})"
            )

        return spread_records

    @property
    def active_rumor_count(self) -> int:
        return len(self._active_rumors)

    @property
    def active_rumors(self) -> list[Rumor]:
        return self._active_rumors.copy()
