"""
记忆管理器 (Memory Manager) — 统一三层记忆的检索与沉淀。

职责：
1. 对外提供统一的 retrieve() / consolidate() 接口
2. 内部协调工作记忆、情景记忆、语义记忆的检索与存储
3. 将对话压缩为记忆条目存入情景记忆
"""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger

from src.models.memory import (
    ConversationTurn,
    MemoryItem,
    MemoryQueryResult,
    MemoryType,
)
from src.memory.working_memory import WorkingMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory


class MemoryManager:
    """
    三层记忆统一管理器。

    每个 NPC 对应一个 MemoryManager 实例，
    内部持有该 NPC 的工作记忆、情景记忆和语义记忆。
    """

    def __init__(self, character_id: str):
        self.character_id = character_id
        self.working = WorkingMemory(character_id)
        self.episodic = EpisodicMemory(character_id)
        self.semantic = SemanticMemory(character_id)
        logger.info(f"[MemoryManager] 初始化完成: character={character_id}")

    def retrieve(self, query: str, player_id: str = "player") -> MemoryQueryResult:
        """
        综合检索三层记忆。

        Parameters
        ----------
        query : 玩家的输入文本
        player_id : 玩家 ID（用于图谱查询）

        Returns
        -------
        MemoryQueryResult : 包含三层记忆检索结果的聚合对象
        """
        # 1. 工作记忆 — 直接取最近对话
        working_memories = self.working.get_recent()

        # 2. 情景记忆 — 向量语义检索
        episodic_memories = self.episodic.retrieve(query)

        # 3. 语义记忆 — 图谱关系查询（从玩家和 NPC 两个视角）
        player_facts = self.semantic.to_facts(player_id, depth=1)
        npc_facts = self.semantic.to_facts(self.character_id, depth=2)

        # 去重合并
        all_facts = list(dict.fromkeys(player_facts + npc_facts))

        player_relations = self.semantic.query_relations(player_id, depth=1)
        npc_relations = self.semantic.query_relations(self.character_id, depth=2)
        all_relations = player_relations + [
            r for r in npc_relations if r not in player_relations
        ]

        result = MemoryQueryResult(
            working_memories=working_memories,
            episodic_memories=episodic_memories,
            semantic_facts=all_facts,
            graph_relations=all_relations,
        )

        logger.debug(
            f"[MemoryManager] 检索完成: "
            f"working={len(working_memories)}, "
            f"episodic={len(episodic_memories)}, "
            f"semantic_facts={len(all_facts)}"
        )
        return result

    def add_turn(self, turn: ConversationTurn) -> None:
        """将一轮对话加入工作记忆。"""
        self.working.add(turn)

    def consolidate(
        self,
        player_input: str,
        npc_response: str,
        emotion_score: float = 5.0,
        importance: float = 5.0,
        related_entities: Optional[list[str]] = None,
    ) -> None:
        """
        记忆沉淀：将本次交互压缩存入情景记忆。

        Parameters
        ----------
        player_input : 玩家说的话
        npc_response : NPC 回复的台词
        emotion_score : 本次交互的情绪强度（由 LLM 评估）
        importance : 重要性评分
        related_entities : 涉及的实体名称
        """
        # 将对话压缩为一条记忆
        content = f"玩家说：{player_input}\n{self.character_id}回复：{npc_response}"

        memory = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            content=content,
            summary=f"与玩家的一次对话",
            emotion_score=emotion_score,
            importance=importance,
            character_id=self.character_id,
            related_entities=related_entities or [],
        )

        self.episodic.store(memory)
        logger.debug(
            f"[MemoryManager] 记忆沉淀完成: emotion={emotion_score}, importance={importance}"
        )

    def init_relationships(self, relationships: list[dict[str, Any]]) -> None:
        """
        从角色配置初始化语义记忆中的关系。

        Parameters
        ----------
        relationships : 来自 CharacterProfile.initial_relationships 的关系列表
        """
        for rel in relationships:
            target_id = rel.get("target_id", "")
            target_name = rel.get("target_name", target_id)

            self.semantic.add_entity(self.character_id, entity_type="npc")
            self.semantic.add_entity(target_id, entity_type="character", name=target_name)
            self.semantic.add_relation(
                source=self.character_id,
                target=target_id,
                relation=rel.get("label", "knows"),
                weight=rel.get("affection", 50) / 100.0,
            )
        logger.info(
            f"[MemoryManager] 初始化关系图谱: {len(relationships)} 条关系"
        )
