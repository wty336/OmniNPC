"""
实体-关系自动抽取器 (Entity Extractor) — 从对话中自动构建知识图谱。

调用 LLM 从对话文本中提取 (subject, relation, object, weight) 四元组，
自动更新 SemanticMemory 中的 NetworkX 图谱。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.llm.client import get_llm_client
from src.memory.semantic_memory import SemanticMemory

# 实体关系抽取 Prompt
ENTITY_EXTRACTION_PROMPT = """请从以下对话中提取实体和关系。

## 对话内容
玩家「{player_name}」说: 「{player_input}」
NPC「{character_name}」回复: 「{npc_response}」

## 已知实体 ID
- 玩家 ID: `{player_id}`
- NPC ID: `{character_id}`

---

请以 JSON 格式输出（不要写其他内容）:
{{
    "entities": [
        {{"id": "实体ID", "type": "person/item/location/event", "name": "显示名称"}}
    ],
    "relations": [
        {{
            "source": "源实体ID",
            "target": "目标实体ID",
            "relation": "关系描述",
            "weight": <float 0-1>
        }}
    ]
}}

注意：
1. 只提取**明确提及**的实体和关系
2. 玩家用 `{player_id}`，NPC 用 `{character_id}`
3. 无新实体/关系就返回空列表"""


class EntityExtractor:
    """实体-关系抽取器，从对话自动更新图谱。"""

    def extract_and_update(
        self,
        player_input: str,
        npc_response: str,
        player_id: str,
        player_name: str,
        character_id: str,
        character_name: str,
        semantic_memory: SemanticMemory,
    ) -> dict[str, Any]:
        """
        从对话中抽取实体和关系，并更新图谱。

        Returns
        -------
        dict : {"entities": [...], "relations": [...]}
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            player_name=player_name,
            player_input=player_input,
            character_name=character_name,
            npc_response=npc_response,
            player_id=player_id,
            character_id=character_id,
        )

        llm = get_llm_client()
        try:
            result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "你是一个知识图谱实体关系抽取器，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            entities = result.get("entities", [])
            relations = result.get("relations", [])

            for entity in entities:
                semantic_memory.add_entity(
                    entity_id=entity.get("id", ""),
                    entity_type=entity.get("type", "unknown"),
                    name=entity.get("name", ""),
                )

            for rel in relations:
                source = rel.get("source", "")
                target = rel.get("target", "")
                relation = rel.get("relation", "")
                weight = max(0.0, min(1.0, float(rel.get("weight", 0.5))))

                if source and target and relation:
                    semantic_memory.add_relation(
                        source=source, target=target,
                        relation=relation, weight=weight,
                    )

            logger.info(
                f"[EntityExtractor] 抽取完成: "
                f"{len(entities)} 实体, {len(relations)} 关系 "
                f"(图谱: {semantic_memory.node_count} 节点, {semantic_memory.edge_count} 边)"
            )
            return {"entities": entities, "relations": relations}

        except Exception as e:
            logger.warning(f"[EntityExtractor] 抽取失败: {e}")
            return {"entities": [], "relations": []}
