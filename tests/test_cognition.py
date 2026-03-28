"""
认知流水线测试（使用 Mock LLM）。
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.models.character import CharacterProfile, Personality, Relationship
from src.models.game_state import GameState
from src.models.memory import MemoryQueryResult, ConversationTurn


def make_test_character() -> CharacterProfile:
    """创建测试用角色。"""
    return CharacterProfile(
        id="test_npc",
        name="测试角色",
        role="测试用 NPC",
        personality=Personality(
            traits=["温和", "善良"],
            speaking_style="温柔体贴",
        ),
        initial_relationships=[
            Relationship(
                target_id="player",
                target_name="玩家",
                affection=60,
                trust=50,
                label="友人",
            )
        ],
        system_prompt="你是一个测试用的 NPC 角色。",
    )


def make_test_game_state() -> GameState:
    """创建测试用游戏状态。"""
    return GameState(session_id="test_session")


class TestSemanticMemory:
    """语义记忆（图谱）测试。"""

    def test_add_and_query_relation(self):
        """测试添加和查询关系。"""
        from src.memory.semantic_memory import SemanticMemory

        sm = SemanticMemory("test_npc")
        sm.add_relation("player", "master", "师徒", weight=0.9)

        relations = sm.query_relations("player", depth=1)
        assert len(relations) >= 1
        assert any(r["relation"] == "师徒" for r in relations)

    def test_to_facts(self):
        """测试转换为自然语言事实。"""
        from src.memory.semantic_memory import SemanticMemory

        sm = SemanticMemory("test_npc")
        sm.add_relation("凌霜", "player", "师姐弟", weight=0.7)

        facts = sm.to_facts("凌霜")
        assert len(facts) == 1
        assert "师姐弟" in facts[0]

    def test_multi_hop_query(self):
        """测试多跳查询。"""
        from src.memory.semantic_memory import SemanticMemory

        sm = SemanticMemory("test_npc")
        sm.add_relation("A", "B", "认识", weight=0.5)
        sm.add_relation("B", "C", "师徒", weight=0.8)

        relations_depth1 = sm.query_relations("A", depth=1)
        relations_depth2 = sm.query_relations("A", depth=2)

        assert len(relations_depth2) > len(relations_depth1)


class TestGameState:
    """游戏状态测试。"""

    def test_update_affection(self):
        """测试好感度修改。"""
        state = make_test_game_state()
        new_val = state.update_affection("npc_1", "player", 15)
        assert new_val == 65  # 默认 50 + 15

    def test_affection_bounds(self):
        """测试好感度上下限。"""
        state = make_test_game_state()
        state.update_affection("npc_1", "player", 200)
        rel = state.get_relationship("npc_1", "player")
        assert rel.affection == 100  # 不超过 100

        state.update_affection("npc_1", "player", -300)
        assert rel.affection == 0  # 不低于 0

    def test_get_or_create_relationship(self):
        """测试自动创建关系。"""
        state = make_test_game_state()
        rel = state.get_relationship("a", "b")
        assert rel.source_id == "a"
        assert rel.target_id == "b"
        assert rel.affection == 50  # 默认值
