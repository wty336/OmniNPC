"""
认知管线编排器 (Cognitive Pipeline) — 基于 LangGraph 重构。

Phase 3 升级:
- 剥离了手写执行管线，全面走向 LangGraph `StateGraph` 构建工作流。
- 引入 `AgentState` 作为每个节点的流转对象。
- 增加条件转移判断。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from loguru import logger
from langgraph.graph import END, START, StateGraph

from src.cognition.action_generator import ActionGenerator
from src.cognition.agent_state import AgentState
from src.cognition.inner_monologue import InnerMonologue
from src.cognition.perception import Perception
from src.memory.emotion_scorer import EmotionScorer
from src.memory.entity_extractor import EntityExtractor
from src.memory.memory_manager import MemoryManager
from src.models.character import CharacterProfile
from src.models.game_state import GameState
from src.models.message import AgentResponse
from src.tools.base import get_tool_registry


class CognitivePipeline:
    """
    基于 LangGraph 的认知流水线引擎。
    管理 感知 -> 反思 -> 决策 -> 执行 -> 记忆沉淀。
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # 实例化各个认知组件
        self.perception = Perception(memory_manager)
        self.inner_monologue = InnerMonologue()
        self.action_generator = ActionGenerator()
        self.emotion_scorer = EmotionScorer()
        self.entity_extractor = EntityExtractor()
        
        # 构建 LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 的节点和连接边。"""
        workflow = StateGraph(AgentState)
        
        # 注册节点
        workflow.add_node("perception", self._node_perception)
        workflow.add_node("monologue", self._node_monologue)
        workflow.add_node("action", self._node_action)
        workflow.add_node("tool_execution", self._node_tool_execution)
        workflow.add_node("consolidation", self._node_consolidation)
        
        # 定义执行边
        workflow.add_edge(START, "perception")
        workflow.add_edge("perception", "monologue")
        workflow.add_edge("monologue", "action")
        
        # 条件路由：基于 action 节点是否产生了工具调用
        workflow.add_conditional_edges(
            "action",
            self._route_after_action,
            {
                "execute_tools": "tool_execution",
                "consolidate": "consolidation"
            }
        )
        
        # 工具执行完后走向记忆沉淀（也可在此处回卷路由到 action 进行多轮思考，暂时线性向下）
        workflow.add_edge("tool_execution", "consolidation")
        workflow.add_edge("consolidation", END)
        
        # 编译计算图
        return workflow.compile()

    # ================= 节点定义 =================

    def _node_perception(self, state: AgentState) -> dict:
        """节点：感知环境与历史记忆"""
        logger.info("[Node] Perception")
        context = self.perception.perceive(
            state["player_input"], 
            state["character"],
            state["game_state"]
        )
        return {"context": context}

    def _node_monologue(self, state: AgentState) -> dict:
        """节点：生成内心独白"""
        logger.info("[Node] Inner Monologue")
        monologue = self.inner_monologue.generate(state["context"])
        return {"monologue": monologue}

    def _node_action(self, state: AgentState) -> dict:
        """节点：生成台词和决定行动"""
        logger.info("[Node] Action Generation")
        response = self.action_generator.generate(
            state["context"], 
            state["monologue"]
        )
        return {
            "dialogue": response.dialogue,
            "tool_calls": response.tool_calls,
        }

    def _node_tool_execution(self, state: AgentState) -> dict:
        """节点：执行工具回调更新系统状态"""
        logger.info("[Node] Tool Execution")
        tool_results = []
        registry = get_tool_registry()
        
        for tool_call in state.get("tool_calls", []):
            if tool_call.tool_name not in registry:
                res = f"工具 {tool_call.tool_name} 不存在"
                tool_call.result = res
                tool_results.append(res)
                logger.warning(f"[Tool Execution] {res}")
                continue
                
            try:
                import inspect
                func = registry[tool_call.tool_name]
                
                # 预填入隐藏状态参数
                kwargs = dict(tool_call.arguments)
                sig = inspect.signature(func)
                if "game_state" in sig.parameters:
                    kwargs["game_state"] = state["game_state"]
                if "character_id" in sig.parameters:
                    kwargs["character_id"] = state["character"].id
                    
                result = func(**kwargs)
                
                # 回填执行结果
                res_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                tool_call.result = res_str
                tool_results.append(f"{tool_call.tool_name} => {res_str}")
                
            except Exception as e:
                res = f"执行报错: {e}"
                tool_call.result = res
                tool_results.append(res)
                logger.error(f"[Tool Execution] {tool_call.tool_name} 失败: {e}")
                
        return {"tool_results": tool_results}

    def _node_consolidation(self, state: AgentState) -> dict:
        """节点：进行情绪打分、关系抽取并打包成情景记忆沉淀"""
        logger.info("[Node] Consolidation")
        
        # 拼装完整的对话结果体供最后打分判断
        response = AgentResponse(
            dialogue=state.get("dialogue", ""),
            inner_monologue=state.get("monologue"),
            tool_calls=state.get("tool_calls", []),
            character_id=state["character"].id,
            metadata={}
        )
        
        # 获取好感度变量供打分参考
        rel = state["game_state"].get_relationship(
            state["game_state"].player.player_id, 
            state["character"].id
        )
        
        # 1. 记忆强化情绪判断
        emotion_data = self.emotion_scorer.score(
            player_input=state["player_input"],
            npc_response=response.dialogue,
            inner_monologue=response.inner_monologue,
            character_name=state["character"].name,
            character_role=state["character"].role,
            affection=rel.affection
        )
        response.metadata["emotion"] = emotion_data
        
        # 2. 从对话中抽取新的关系保存到 SemanticMemory 图谱中
        self.entity_extractor.extract_and_update(
            player_input=state["player_input"],
            npc_response=response.dialogue,
            player_id=state["game_state"].player.player_id,
            player_name=state["game_state"].player.name,
            character_id=state["character"].id,
            character_name=state["character"].name,
            semantic_memory=self.memory_manager.semantic
        )
        
        # 3. 产生最终情境记忆快照并写入 ChromaDB 和 滑动窗口
        self.memory_manager.consolidate(
            player_input=state["player_input"],
            npc_response=response.dialogue,
            emotion_score=emotion_data["emotion_score"],
            importance=emotion_data["importance"]
        )
        
        return {
            "final_response": response,
            "emotion_score": emotion_data["emotion_score"],
            "importance": emotion_data["importance"]
        }

    # ================= 边缘路由判断 =================
    
    def _route_after_action(self, state: AgentState) -> str:
        """条件边：判断 Action 阶段后是否出现了工具调用参数。"""
        if state.get("tool_calls") and len(state["tool_calls"]) > 0:
            return "execute_tools"
        return "consolidate"

    # ================= 暴露接口 =================

    def run(
        self,
        player_input: str,
        character: CharacterProfile,
        game_state: GameState,
    ) -> AgentResponse:
        """
        触发编排网络的执行输入点。
        返回封装好的响应体。
        """
        # 初始状态载入
        initial_state = {
            "player_input": player_input,
            "character": character,
            "game_state": game_state,
            "tool_calls": [],
            "tool_results": [],
            "metadata": {}
        }
        
        # 调度 LangGraph 执行，传入 State
        final_state = self.graph.invoke(initial_state)
        
        # 取消返回 State 对象，统一为框架预设好的 response 发送
        if "final_response" in final_state and final_state["final_response"]:
            return final_state["final_response"]
            
        raise RuntimeError("LangGraph execution failed to produce a final response.")
