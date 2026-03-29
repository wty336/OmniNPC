"""
Phase 2 新模块测试：遗忘曲线、流言传播、Tick 引擎。
"""

import math
import sys
import time
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch
import pytest

from src.models.memory import MemoryItem, MemoryType

loguru_stub = ModuleType("loguru")
loguru_stub.logger = SimpleNamespace(
    info=lambda *args, **kwargs: None,
    debug=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
)
sys.modules.setdefault("loguru", loguru_stub)


class TestMemoryDecay:
    """艾宾浩斯遗忘曲线测试。"""

    def test_compute_stability(self):
        """测试稳定性计算。"""
        from src.memory.decay import MemoryDecay

        # 低情绪低重要性 → 低稳定性
        s_low = MemoryDecay.compute_stability(2.0, 2.0)
        # 高情绪高重要性 → 高稳定性
        s_high = MemoryDecay.compute_stability(9.0, 9.0)
        assert s_high > s_low * 3  # 高情绪的稳定性至少是低情绪的 3 倍

    def test_retention_fresh_memory(self):
        """测试新鲜记忆的保留率应接近 1.0。"""
        from src.memory.decay import MemoryDecay

        memory = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            content="测试记忆",
            emotion_score=5.0,
            importance=5.0,
        )
        retention = MemoryDecay.compute_retention(memory)
        assert retention > 0.99  # 刚创建的记忆几乎不衰减

    def test_retention_old_memory(self):
        """测试旧记忆的保留率低于新记忆。"""
        from src.memory.decay import MemoryDecay

        old_memory = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            content="旧记忆",
            emotion_score=3.0,
            importance=3.0,
            created_at=time.time() - 72 * 3600,  # 72 小时前
        )
        retention = MemoryDecay.compute_retention(old_memory)
        assert retention < 0.8  # 低情绪的旧记忆应有所衰减

    def test_permanent_memory(self):
        """测试永久记忆（高情绪+高重要性）不衰减。"""
        from src.memory.decay import MemoryDecay

        permanent_memory = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            content="承诺永远守护",
            emotion_score=9.0,
            importance=9.0,
            created_at=time.time() - 720 * 3600,  # 30 天前
        )
        retention = MemoryDecay.compute_retention(permanent_memory)
        assert retention == 1.0  # 永久记忆不衰减

    def test_apply_decay_batch(self):
        """测试批量衰减和分类。"""
        from src.memory.decay import MemoryDecay

        memories = [
            MemoryItem(
                memory_type=MemoryType.EPISODIC,
                content="新鲜记忆",
                emotion_score=5.0,
                importance=5.0,
            ),
            MemoryItem(
                memory_type=MemoryType.EPISODIC,
                content="极旧的日常闲聊",
                emotion_score=1.0,
                importance=1.0,
                created_at=time.time() - 720 * 3600,  # 30 天前
            ),
        ]
        active, forgotten = MemoryDecay.apply_decay(memories)
        assert len(active) >= 1
        assert len(forgotten) >= 1
        assert active[0].decay_weight > 0.9
        assert forgotten[0].decay_weight < 0.05

    def test_weighted_score(self):
        """测试加权检索得分。"""
        from src.memory.decay import MemoryDecay

        memory = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            content="test",
            decay_weight=0.5,
        )
        score = MemoryDecay.weighted_score(memory, similarity=0.8)
        assert abs(score - 0.4) < 0.01  # 0.8 * 0.5 = 0.4


class TestRumorSpreader:
    """流言传播测试。"""

    def test_create_rumor(self):
        """测试创建流言。"""
        from src.sandbox.rumor_spreader import RumorSpreader

        spreader = RumorSpreader()
        rumor = spreader.create_rumor(
            content="玩家打碎了师傅的花瓶",
            source_npc="tsundere_sister",
            original_event="花瓶事件",
        )
        assert rumor.credibility == 1.0
        assert "tsundere_sister" in rumor.heard_by
        assert spreader.active_rumor_count == 1

    def test_rumor_degrade(self):
        """测试流言可信度递减。"""
        from src.sandbox.rumor_spreader import Rumor

        rumor = Rumor(
            content="test",
            source_npc="npc_a",
            original_event="event",
        )
        assert rumor.credibility == 1.0
        rumor.degrade(factor=0.85)
        assert abs(rumor.credibility - 0.85) < 0.01
        rumor.degrade(factor=0.85)
        assert abs(rumor.credibility - 0.7225) < 0.01

    def test_spread_tick_supports_memory_sink_without_episodic_store(self):
        """测试流言传播支持更轻量的记忆写入口。"""
        from src.sandbox.rumor_spreader import RumorSpreader

        class MemorySink:
            def __init__(self):
                self.memories = []

            def store_memory_item(self, memory):
                self.memories.append(memory)

        sink = MemorySink()
        spreader = RumorSpreader()
        spreader.create_rumor(
            content="玩家昨夜偷偷翻墙下山",
            source_npc="npc_a",
            original_event="翻墙事件",
        )

        with patch("src.sandbox.rumor_spreader.random.random", return_value=0.0), patch(
            "src.sandbox.rumor_spreader.random.choice",
            return_value="npc_b",
        ):
            records = spreader.spread_tick(
                all_npc_ids=["npc_a", "npc_b"],
                get_memory_manager=lambda npc_id: sink if npc_id == "npc_b" else None,
            )

        assert len(records) == 1
        assert sink.memories[0].metadata["type"] == "rumor"
        assert sink.memories[0].character_id == "npc_b"


class TestTickEngine:
    """Tick 引擎测试。"""

    def test_manual_tick(self):
        """测试手动执行 Tick。"""
        from src.sandbox.tick_engine import TickEngine

        engine = TickEngine()
        result = engine.tick()
        assert result["tick_id"] == 1
        assert "decay_results" in result

        result2 = engine.tick()
        assert result2["tick_id"] == 2
        assert engine.tick_count == 2

    def test_tick_uses_public_memory_manager_iterator_when_available(self):
        """测试 TickEngine 优先使用公开的记忆迭代接口。"""
        from src.sandbox.tick_engine import TickEngine

        fake_engine = SimpleNamespace(
            iter_memory_managers=lambda: [("npc_a", SimpleNamespace(episodic=object()))]
        )

        engine = TickEngine()
        result = engine.tick(engine=fake_engine)

        assert result["decay_results"] == {"npc_a": {"status": "updated"}}
