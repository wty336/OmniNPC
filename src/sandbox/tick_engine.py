"""
离线 Tick 引擎 — 驱动 NPC 自主行为和记忆衰减。

使用 APScheduler 定时触发 Tick，每个 Tick 中：
1. 执行记忆衰减（遗忘曲线）
2. 检查 NPC 自主行为条件
3. 传播流言（NPC 间信息扩散）
4. 更新世界状态

这是 NPC"活"起来的关键——即使玩家不在线，NPC 也在经历时间流逝。
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.memory.decay import MemoryDecay


class TickEngine:
    """
    离线 Tick 引擎。

    管理全局时间推进和 NPC 自主行为调度。
    """

    def __init__(self):
        self._tick_count: int = 0
        self._last_tick_time: float = time.time()
        self._running: bool = False
        self._scheduler = None
        logger.info("[TickEngine] 初始化完成")

    def start(self, interval_seconds: int = 300):
        """
        启动定时 Tick。

        Parameters
        ----------
        interval_seconds : Tick 间隔（秒），默认 5 分钟
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            logger.warning("[TickEngine] APScheduler 未安装，使用手动 Tick 模式")
            return

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self.tick,
            trigger="interval",
            seconds=interval_seconds,
            id="omni_npc_tick",
        )
        self._scheduler.start()
        self._running = True
        logger.info(f"[TickEngine] 已启动，间隔 {interval_seconds}s")

    def stop(self):
        """停止定时 Tick。"""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("[TickEngine] 已停止")

    def tick(
        self,
        engine: Optional[object] = None,
    ) -> dict:
        """
        执行一次 Tick。

        Parameters
        ----------
        engine : NPCEngine 实例（可选，用于访问角色和记忆）

        Returns
        -------
        dict : Tick 执行结果
        """
        self._tick_count += 1
        now = time.time()
        elapsed = now - self._last_tick_time
        self._last_tick_time = now

        logger.info(
            f"[TickEngine] ⏰ Tick #{self._tick_count} "
            f"(距上次 {elapsed:.0f}s)"
        )

        results = {
            "tick_id": self._tick_count,
            "elapsed_seconds": elapsed,
            "decay_results": {},
            "autonomous_actions": [],
            "rumor_spreads": [],
        }

        # 1. 记忆衰减
        if engine:
            results["decay_results"] = self._process_memory_decay(engine)

        # 2. NPC 自主行为（Phase 2 雏形）
        # TODO: 基于 NPC 当前状态和性格触发自主行为

        # 3. 流言传播
        # 由 RumorSpreader 处理，通过 engine 调用

        logger.info(
            f"[TickEngine] Tick #{self._tick_count} 完成: "
            f"衰减={len(results['decay_results'])} 个 NPC"
        )
        return results

    def _process_memory_decay(self, engine) -> dict:
        """处理所有 NPC 的记忆衰减。"""
        decay_results = {}

        for character_id, memory_manager in engine._memory_managers.items():
            # 获取所有情景记忆并应用衰减
            # Phase 2: 只做衰减权重更新，不物理删除
            episodic = memory_manager.episodic
            try:
                # ChromaDB doesn't support batch update easily,
                # so we update decay_weight on retrieval
                decay_results[character_id] = {"status": "updated"}
                logger.debug(
                    f"[TickEngine] 记忆衰减: {character_id}"
                )
            except Exception as e:
                logger.warning(f"[TickEngine] 记忆衰减失败 ({character_id}): {e}")
                decay_results[character_id] = {"status": "error", "error": str(e)}

        return decay_results

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def is_running(self) -> bool:
        return self._running
