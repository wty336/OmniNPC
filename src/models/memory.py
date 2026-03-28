"""记忆数据模型 — 工作记忆 / 情景记忆 / 语义记忆的统一数据结构。"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型枚举。"""

    WORKING = "working"      # 短期工作记忆（最近对话）
    EPISODIC = "episodic"    # 情景记忆（事件片段）
    SEMANTIC = "semantic"    # 语义记忆（知识图谱中的事实）


class MemoryItem(BaseModel):
    """
    单条记忆条目。

    无论记忆类型如何，都用统一结构存储，
    通过 metadata 字段携带类型特定的附加信息。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = Field(description="记忆文本内容")
    summary: str = Field(default="", description="记忆摘要（可由 LLM 提炼）")

    # 时间信息
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)

    # 情绪与重要性（用于遗忘曲线）
    emotion_score: float = Field(
        default=5.0, ge=0, le=10,
        description="情绪强度 0-10，越高越难遗忘",
    )
    importance: float = Field(
        default=5.0, ge=0, le=10,
        description="重要性评分 0-10",
    )
    decay_weight: float = Field(
        default=1.0, ge=0, le=1,
        description="当前衰减后的权重，0=完全遗忘，1=完全清晰",
    )

    # 关联信息
    character_id: str = Field(default="", description="所属 NPC ID")
    related_entities: list[str] = Field(
        default_factory=list,
        description="相关实体名称列表",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据",
    )


class ConversationTurn(BaseModel):
    """单轮对话记录（工作记忆的基本单元）。"""

    role: str = Field(description="发言者角色：'player' / 'npc' / 'system'")
    speaker_name: str = Field(default="", description="发言者名称")
    content: str = Field(description="对话文本")
    timestamp: float = Field(default_factory=time.time)
    inner_monologue: Optional[str] = Field(
        default=None,
        description="NPC 内心独白（仅 NPC 发言时有值，不对玩家展示）",
    )


class MemoryQueryResult(BaseModel):
    """记忆检索结果。"""

    working_memories: list[ConversationTurn] = Field(default_factory=list)
    episodic_memories: list[MemoryItem] = Field(default_factory=list)
    semantic_facts: list[str] = Field(
        default_factory=list,
        description="语义记忆中检索到的事实描述",
    )
    graph_relations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="图谱关系 [{source, relation, target, weight}]",
    )
