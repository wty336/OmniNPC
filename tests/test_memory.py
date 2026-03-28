"""
记忆系统单元测试。
"""

import pytest

from src.models.memory import ConversationTurn, MemoryItem, MemoryType
from src.memory.working_memory import WorkingMemory


class TestWorkingMemory:
    """工作记忆测试。"""

    def test_add_and_retrieve(self):
        """测试添加和获取对话。"""
        wm = WorkingMemory("test_npc", window_size=5)
        turn = ConversationTurn(
            role="player", speaker_name="玩家", content="你好"
        )
        wm.add(turn)
        assert wm.size == 1
        assert wm.get_recent()[0].content == "你好"

    def test_sliding_window(self):
        """测试滑动窗口淘汰机制。"""
        wm = WorkingMemory("test_npc", window_size=3)
        for i in range(5):
            wm.add(ConversationTurn(
                role="player", speaker_name="玩家", content=f"消息{i}"
            ))
        assert wm.size == 3
        contents = [t.content for t in wm.get_recent()]
        assert contents == ["消息2", "消息3", "消息4"]

    def test_to_messages(self):
        """测试转换为 LLM 消息格式。"""
        wm = WorkingMemory("test_npc", window_size=10)
        wm.add(ConversationTurn(role="player", speaker_name="玩家", content="你好"))
        wm.add(ConversationTurn(role="npc", speaker_name="凌霜", content="哼，什么事？"))

        messages = wm.to_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "你好"}
        assert messages[1] == {"role": "assistant", "content": "哼，什么事？"}

    def test_clear(self):
        """测试清空工作记忆。"""
        wm = WorkingMemory("test_npc", window_size=5)
        wm.add(ConversationTurn(role="player", speaker_name="玩家", content="test"))
        wm.clear()
        assert wm.size == 0


class TestMemoryItem:
    """记忆数据模型测试。"""

    def test_create_memory(self):
        """测试创建记忆条目。"""
        memory = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            content="玩家打碎了师傅的花瓶",
            emotion_score=8.5,
            importance=9.0,
            character_id="tsundere_sister",
            related_entities=["玩家", "师傅", "花瓶"],
        )
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.emotion_score == 8.5
        assert "花瓶" in memory.related_entities
        assert 0 <= memory.decay_weight <= 1
