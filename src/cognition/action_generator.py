"""
决策与行动生成模块 (Action Generator) — 认知管线的第三步。

职责：
1. 结合感知上下文 + 内心独白，生成对外台词
2. 决定是否调用工具（Function Calling）修改游戏状态
3. 如果触发了工具调用，将工具结果反馈给 LLM 再生成最终台词
4. 输出结构化的 AgentResponse
"""

from __future__ import annotations

import json

from loguru import logger

from src.adapters.llm.ark_adapter import ArkModelAdapter
from src.adapters.llm.base import ModelRequest
from src.cognition.action_planner import (
    ACTION_SYSTEM_PROMPT as _ACTION_SYSTEM_PROMPT,
    ActionPlanner,
    build_action_messages,
)
from src.cognition.perception import PerceptionContext
from src.models.message import AgentResponse, ToolCallResult
from src.tools.base import get_tool_definitions, get_tool_registry

ACTION_SYSTEM_PROMPT = _ACTION_SYSTEM_PROMPT


class ActionGenerator:
    """
    行动生成器。

    结合内心独白和记忆上下文，生成：
    - 对外台词（玩家可见）
    - 工具调用（修改游戏状态）

    当 LLM 返回工具调用时，会执行工具并将结果反馈给 LLM，
    让 LLM 根据工具执行结果生成最终台词（二次调用）。
    """

    def __init__(self, model_adapter: ArkModelAdapter | None = None):
        self._model = model_adapter or ArkModelAdapter()
        self._planner = ActionPlanner(model_adapter=self._model)

    def generate(
        self,
        context: PerceptionContext,
        inner_monologue: str,
    ) -> AgentResponse:
        """
        生成行动（台词 + 工具调用）。

        Parameters
        ----------
        context : 感知上下文
        inner_monologue : 内心独白文本

        Returns
        -------
        AgentResponse : 包含台词和工具调用的完整响应
        """
        character = context.character
        messages = build_action_messages(context, inner_monologue)
        tools = get_tool_definitions()

        logger.debug(f"[ActionGenerator] 第一次调用 LLM... tools={len(tools)} 个")
        plan = self._planner.plan(context, inner_monologue, tools=tools)

        dialogue = plan.dialogue
        tool_call_results = []

        # ── 处理工具调用 ──
        if plan.tool_calls:
            tool_registry = get_tool_registry()
            assistant_tool_calls = []
            tool_messages = []

            for index, tool_call in enumerate(plan.tool_calls, start=1):
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {}

                tool_call_id = tool_call.get("id") or f"call-{index}"
                assistant_tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parsed_arguments, ensure_ascii=False),
                        },
                    }
                )

                logger.info(f"[ActionGenerator] 工具调用: {tool_name}({parsed_arguments})")

                tool_func = tool_registry.get(tool_name)
                if tool_func:
                    try:
                        tool_result_data = tool_func(
                            game_state=context.game_state, **parsed_arguments
                        )
                        tool_call_results.append(ToolCallResult(
                            tool_name=tool_name,
                            arguments=parsed_arguments,
                            result=tool_result_data,
                            success=True,
                        ))
                    except Exception as e:
                        logger.error(f"工具执行失败: {tool_name} -> {e}")
                        tool_result_data = {"error": str(e)}
                        tool_call_results.append(ToolCallResult(
                            tool_name=tool_name,
                            arguments=parsed_arguments,
                            success=False,
                            error=str(e),
                        ))
                else:
                    logger.warning(f"未知工具: {tool_name}")
                    tool_result_data = {"error": f"未知工具: {tool_name}"}
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result_data, ensure_ascii=False),
                })
            messages.append({
                "role": "assistant",
                "content": plan.dialogue or "",
                "tool_calls": assistant_tool_calls,
            })
            messages.extend(tool_messages)

            # ── 第二次 LLM 调用：根据工具结果生成最终台词 ──
            logger.debug("[ActionGenerator] 第二次调用 LLM（基于工具结果生成台词）...")
            second_result = self._model.complete(
                ModelRequest(
                    purpose="respond",
                    messages=messages,
                    temperature=0.8,
                )
            )
            dialogue = second_result.content or ""

        response = AgentResponse(
            dialogue=dialogue,
            inner_monologue=inner_monologue,
            tool_calls=tool_call_results,
            character_id=character.id,
            character_name=character.name,
        )

        logger.info(
            f"[ActionGenerator] 生成完成: 台词='{dialogue[:50]}...', "
            f"工具调用={len(tool_call_results)} 个"
        )
        return response
