"""游戏状态数据模型 — 玩家 / 世界 / 关系等运行时状态。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PlayerState(BaseModel):
    """玩家当前状态。"""

    player_id: str = "player_default"
    name: str = "旅行者"
    location: str = "玄天宗·大殿"
    inventory: list[str] = Field(default_factory=list, description="持有道具")
    attributes: dict[str, float] = Field(
        default_factory=lambda: {"health": 100, "mana": 100},
        description="自定义属性（生命、法力等）",
    )


class NPCState(BaseModel):
    """NPC 运行时状态（与角色档案区分：档案是静态人设，这里是动态数值）。"""

    character_id: str
    location: str = ""
    mood: str = "neutral"
    current_activity: str = "idle"
    custom_flags: dict[str, Any] = Field(default_factory=dict)


class RelationshipState(BaseModel):
    """关系运行时状态。"""

    source_id: str
    target_id: str
    affection: float = 50.0
    trust: float = 50.0
    label: str = "neutral"
    events_count: int = 0


class GameState(BaseModel):
    """
    完整游戏状态快照。

    这是一个可序列化的全局状态对象，
    引擎通过 Function Calling 修改此状态来实现"言出法随"。
    """

    session_id: str = "default_session"
    player: PlayerState = Field(default_factory=PlayerState)
    npcs: dict[str, NPCState] = Field(default_factory=dict, description="NPC ID -> 运行时状态")
    relationships: dict[str, RelationshipState] = Field(
        default_factory=dict,
        description="'sourceId::targetId' -> 关系状态",
    )
    world_flags: dict[str, Any] = Field(
        default_factory=dict,
        description="全局事件标记（如 '花瓶是否碎了'）",
    )
    tick_count: int = Field(default=0, description="游戏世界推进的 tick 计数")

    # ---- 便捷方法 ----

    def get_relationship(self, source_id: str, target_id: str) -> RelationshipState:
        """获取或创建两个实体之间的关系。"""
        key = f"{source_id}::{target_id}"
        if key not in self.relationships:
            self.relationships[key] = RelationshipState(
                source_id=source_id, target_id=target_id
            )
        return self.relationships[key]

    def update_affection(self, source_id: str, target_id: str, delta: float) -> float:
        """修改好感度并返回新值。"""
        rel = self.get_relationship(source_id, target_id)
        rel.affection = max(0, min(100, rel.affection + delta))
        return rel.affection
