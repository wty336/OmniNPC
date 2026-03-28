"""
语义记忆 (Semantic Memory) — 基于知识图谱的事实记忆。

Phase 1 使用 NetworkX 轻量实现，存储实体之间的关系。
Phase 3 将升级为 Neo4j GraphRAG。
"""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx
from loguru import logger


class SemanticMemory:
    """
    基于 NetworkX 的语义记忆（知识图谱）。

    存储实体（人物、物品、地点）之间的关系，
    支持关系查询和多跳邻域检索。
    """

    def __init__(self, character_id: str):
        self.character_id = character_id
        self._graph = nx.DiGraph()
        logger.debug(f"[SemanticMemory] 初始化: character={character_id}")

    def add_entity(self, entity_id: str, entity_type: str = "unknown", **attrs) -> None:
        """添加实体节点。"""
        self._graph.add_node(entity_id, entity_type=entity_type, **attrs)

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        **attrs,
    ) -> None:
        """
        添加或更新关系边。

        Parameters
        ----------
        source : 源实体 ID
        target : 目标实体 ID
        relation : 关系描述（如 "师徒"、"打碎花瓶"）
        weight : 关系权重
        """
        # 自动创建节点（如果不存在）
        if source not in self._graph:
            self._graph.add_node(source)
        if target not in self._graph:
            self._graph.add_node(target)

        if self._graph.has_edge(source, target):
            # 更新已有关系
            self._graph[source][target]["relation"] = relation
            self._graph[source][target]["weight"] = weight
            self._graph[source][target].update(attrs)
        else:
            self._graph.add_edge(
                source, target, relation=relation, weight=weight, **attrs
            )

        logger.debug(f"[SemanticMemory] 关系: {source} --({relation})--> {target} [w={weight}]")

    def query_relations(
        self, entity_id: str, depth: int = 1
    ) -> list[dict[str, Any]]:
        """
        查询实体的关系网络。

        Parameters
        ----------
        entity_id : 查询的实体 ID
        depth : 搜索深度（1=直接关系，2=包含二跳关系）

        Returns
        -------
        list[dict] : [{"source": ..., "target": ..., "relation": ..., "weight": ...}]
        """
        if entity_id not in self._graph:
            return []

        relations = []
        visited = set()

        def _dfs(node: str, current_depth: int):
            if current_depth > depth or node in visited:
                return
            visited.add(node)

            # 出边（node -> neighbor）
            for neighbor in self._graph.successors(node):
                edge = self._graph[node][neighbor]
                relations.append({
                    "source": node,
                    "target": neighbor,
                    "relation": edge.get("relation", ""),
                    "weight": edge.get("weight", 1.0),
                })
                _dfs(neighbor, current_depth + 1)

            # 入边（neighbor -> node）
            for neighbor in self._graph.predecessors(node):
                edge = self._graph[neighbor][node]
                relations.append({
                    "source": neighbor,
                    "target": node,
                    "relation": edge.get("relation", ""),
                    "weight": edge.get("weight", 1.0),
                })
                if current_depth + 1 <= depth:
                    _dfs(neighbor, current_depth + 1)

        _dfs(entity_id, 1)
        return relations

    def get_entity_info(self, entity_id: str) -> Optional[dict[str, Any]]:
        """获取实体的属性信息。"""
        if entity_id in self._graph:
            return dict(self._graph.nodes[entity_id])
        return None

    def to_facts(self, entity_id: str, depth: int = 1) -> list[str]:
        """
        将实体的关系网络转换为自然语言事实列表。

        Returns
        -------
        list[str] : 如 ["玩家 与 师姐 关系为 师妹 (权重: 0.8)"]
        """
        relations = self.query_relations(entity_id, depth)
        facts = []
        for r in relations:
            facts.append(
                f"{r['source']} 与 {r['target']} 的关系为「{r['relation']}」(权重: {r['weight']:.1f})"
            )
        return facts

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
