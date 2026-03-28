"""
情景记忆 (Episodic Memory) — 基于向量检索的事件记忆。

将每次重要交互压缩为记忆条目，存入 ChromaDB 向量数据库。
检索时根据语义相似度 + 情绪权重 + 时间衰减进行综合排序。
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from loguru import logger

from config.settings import settings
from src.models.memory import MemoryItem, MemoryType


class EpisodicMemory:
    """
    基于 ChromaDB 的情景记忆。

    每个 NPC 对应 ChromaDB 中的一个 collection。
    支持语义检索和元数据过滤。
    """

    def __init__(self, character_id: str):
        self.character_id = character_id
        self._collection = self._get_or_create_collection()
        logger.debug(f"[EpisodicMemory] 初始化: character={character_id}")

    def _get_or_create_collection(self):
        """获取或创建 ChromaDB collection。"""
        import chromadb

        client = chromadb.PersistentClient(path=settings.memory.chroma_persist_dir)
        collection_name = f"episodic_{self.character_id}"
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, memory: MemoryItem) -> None:
        """存储一条情景记忆。"""
        self._collection.add(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[
                {
                    "character_id": self.character_id,
                    "memory_type": memory.memory_type.value,
                    "emotion_score": memory.emotion_score,
                    "importance": memory.importance,
                    "decay_weight": memory.decay_weight,
                    "created_at": memory.created_at,
                    "summary": memory.summary,
                    "related_entities": ",".join(memory.related_entities),
                    "extra_metadata": json.dumps(memory.metadata, ensure_ascii=False) if memory.metadata else "{}",
                }
            ],
        )
        logger.debug(
            f"[EpisodicMemory] 存储记忆: id={memory.id[:8]}..., "
            f"emotion={memory.emotion_score}, importance={memory.importance}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> list[MemoryItem]:
        """
        语义检索相关记忆。

        Parameters
        ----------
        query : 查询文本
        top_k : 返回条数
        min_importance : 最低重要性阈值

        Returns
        -------
        list[MemoryItem] : 按相关性排序的记忆列表
        """
        k = top_k or settings.memory.episodic_top_k
        where_filter = None
        if min_importance > 0:
            where_filter = {"importance": {"$gte": min_importance}}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter,
            )
        except Exception as e:
            logger.warning(f"[EpisodicMemory] 检索失败: {e}")
            return []

        memories = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                related = meta.get("related_entities", "")
                # 反序列化扩展 metadata（含流言标记等）
                extra_meta = {}
                if meta.get("extra_metadata"):
                    try:
                        extra_meta = json.loads(meta["extra_metadata"])
                    except json.JSONDecodeError:
                        pass

                memory = MemoryItem(
                    id=results["ids"][0][i],
                    memory_type=MemoryType.EPISODIC,
                    content=doc,
                    summary=meta.get("summary", ""),
                    emotion_score=meta.get("emotion_score", 5.0),
                    importance=meta.get("importance", 5.0),
                    decay_weight=meta.get("decay_weight", 1.0),
                    created_at=meta.get("created_at", 0),
                    character_id=self.character_id,
                    related_entities=related.split(",") if related else [],
                    metadata=extra_meta,
                )
                memories.append(memory)

        logger.debug(f"[EpisodicMemory] 检索到 {len(memories)} 条记忆 (query: {query[:30]}...)")
        return memories

    def count(self) -> int:
        """返回存储的记忆数量。"""
        return self._collection.count()
