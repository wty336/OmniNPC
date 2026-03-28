"""
Agent State 定义 — 供 LangGraph 认知图谱使用。
用于在各个认知节点（Perception -> Monologue -> Action -> Consolidation）之间流转核心状态。
"""

from typing import Annotated, Any, Optional, TypedDict
import operator

from src.models.character import CharacterProfile
from src.models.game_state import GameState
from src.models.message import AgentResponse, ToolCallResult


class AgentState(TypedDict):
    """
    LangGraph 认知流中的核心上下文状态对象。
    它贯穿于每个状态节点的执行之中。
    """
    
    # 基础输入与上下文
    player_input: str
    character: CharacterProfile
    game_state: GameState
    
    # 获取到的相关记忆上下文
    # context 是结构化的字符串，含有历史对话、情景、图谱
    context: str 
    
    # 生成的中间推理
    monologue: Optional[str]
    
    # 生成的台词与选择的动作
    dialogue: Optional[str]
    tool_calls: list[ToolCallResult]
    
    # 工具调用的结果（List 采用 add 语义合并，适合多轮工具调用循环累加）
    tool_results: Annotated[list[str], operator.add]
    
    # 状态评分与元信息
    emotion_score: float
    importance: float
    metadata: dict[str, Any]
    
    # 最终结果输出
    final_response: Optional[AgentResponse]
