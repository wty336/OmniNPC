"""
艾宾浩斯遗忘曲线 (Ebbinghaus Forgetting Curve) — 记忆衰减引擎。

核心公式: R = e^(-t / S)
  - R: 记忆保留率 (0~1)
  - t: 距记忆创建的时间（小时）
  - S: 记忆稳定性，由情绪强度和重要性决定

高情绪 / 高重要性的记忆衰减极慢（如背叛、救命之恩）。
日常闲聊衰减很快，最终被物理删除。
"""

from __future__ import annotations

import math
import time
from typing import Optional

from loguru import logger

from src.models.memory import MemoryItem


class MemoryDecay:
    """
    记忆衰减计算器。

    控制每条记忆的保留权重，
    并提供批量衰减和过期记忆清理。
    """

    # 基础稳定性常数（小时）：emotion=5, importance=5 时约 48h 衰减到 50%
    BASE_STABILITY: float = 48.0

    # 情绪/重要性的稳定性放大系数
    EMOTION_FACTOR: float = 3.0     # emotion=10 → stability x4
    IMPORTANCE_FACTOR: float = 2.0  # importance=10 → stability x3

    # 低于此阈值的记忆将被标记为可删除
    FORGET_THRESHOLD: float = 0.05

    # 永久记忆阈值：情绪 + 重要性 >= 此值的记忆不会衰减
    PERMANENT_THRESHOLD: float = 17.0  # emotion + importance >= 17 (如 9+8)

    @classmethod
    def compute_stability(cls, emotion_score: float, importance: float) -> float:
        """
        计算记忆稳定性 S。

        稳定性越高，记忆衰减越慢。
        S = BASE * (1 + emotion_factor) * (1 + importance_factor)
        """
        emotion_bonus = (emotion_score / 10.0) * cls.EMOTION_FACTOR
        importance_bonus = (importance / 10.0) * cls.IMPORTANCE_FACTOR
        stability = cls.BASE_STABILITY * (1 + emotion_bonus) * (1 + importance_bonus)
        return stability

    @classmethod
    def compute_retention(
        cls,
        memory: MemoryItem,
        current_time: Optional[float] = None,
    ) -> float:
        """
        计算单条记忆的当前保留率。

        Parameters
        ----------
        memory : 记忆条目
        current_time : 当前时间戳（默认 time.time()）

        Returns
        -------
        float : 保留率 0~1
        """
        now = current_time or time.time()
        elapsed_hours = (now - memory.created_at) / 3600.0

        # 永久记忆不衰减
        if memory.emotion_score + memory.importance >= cls.PERMANENT_THRESHOLD:
            return 1.0

        stability = cls.compute_stability(memory.emotion_score, memory.importance)
        retention = math.exp(-elapsed_hours / stability)
        return max(0.0, min(1.0, retention))

    @classmethod
    def apply_decay(
        cls,
        memories: list[MemoryItem],
        current_time: Optional[float] = None,
    ) -> tuple[list[MemoryItem], list[MemoryItem]]:
        """
        批量应用衰减，更新所有记忆的 decay_weight。

        Returns
        -------
        (active, forgotten) : 活跃记忆列表 和 应被遗忘的记忆列表
        """
        now = current_time or time.time()
        active = []
        forgotten = []

        for memory in memories:
            retention = cls.compute_retention(memory, now)
            memory.decay_weight = retention

            if retention <= cls.FORGET_THRESHOLD:
                forgotten.append(memory)
            else:
                active.append(memory)

        if forgotten:
            logger.info(
                f"[MemoryDecay] 衰减结果: "
                f"{len(active)} 条活跃, {len(forgotten)} 条遗忘"
            )

        return active, forgotten

    @classmethod
    def weighted_score(cls, memory: MemoryItem, similarity: float) -> float:
        """
        计算记忆检索的加权得分：语义相似度 × 衰减权重。

        用于对检索结果重排序，衰减后的记忆排名降低。
        """
        return similarity * memory.decay_weight
