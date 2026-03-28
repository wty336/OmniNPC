"""NPC 角色定义 — 人设、性格、关系等数据模型。"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Personality(BaseModel):
    """角色性格描述。"""

    traits: list[str] = Field(default_factory=list, description="性格特征列表，如 ['傲娇', '嘴硬心软']")
    speaking_style: str = Field(default="", description="说话风格描述")
    values: list[str] = Field(default_factory=list, description="核心价值观")


class Relationship(BaseModel):
    """单条关系数据。"""

    target_id: str = Field(description="关系目标的 ID（玩家或其他 NPC）")
    target_name: str = Field(default="", description="目标名称")
    affection: float = Field(default=50.0, ge=0, le=100, description="好感度 0-100")
    trust: float = Field(default=50.0, ge=0, le=100, description="信任度 0-100")
    label: str = Field(default="neutral", description="关系标签：ally / neutral / hostile / romantic 等")
    description: str = Field(default="", description="关系详细描述（可选，注入 NPC 认知上下文）")


class CharacterProfile(BaseModel):
    """
    NPC 角色完整档案。
    对应 data/characters/<name>.yaml 中的配置。
    """

    id: str = Field(description="唯一标识符")
    name: str = Field(description="角色名称")
    role: str = Field(default="", description="角色定位，如 '傲娇师姐'")
    backstory: str = Field(default="", description="角色背景故事")
    personality: Personality = Field(default_factory=Personality)
    initial_relationships: list[Relationship] = Field(
        default_factory=list,
        description="初始关系列表",
    )
    location: str = Field(default="未知", description="NPC 所在地点")
    system_prompt: str = Field(
        default="",
        description="注入 LLM 的系统提示词（角色扮演指令）",
    )
    avatar: Optional[str] = Field(default=None, description="头像路径或 URL")

    @classmethod
    def from_yaml(cls, path: str) -> "CharacterProfile":
        """从 YAML 文件加载角色。"""
        import yaml
        from pathlib import Path

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
