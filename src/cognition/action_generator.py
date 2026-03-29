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
from src.cognition.perception import PerceptionContext
from src.models.message import AgentResponse, ToolCallResult
from src.tools.base import get_tool_definitions, get_tool_registry

# 行动生成的系统 Prompt
ACTION_SYSTEM_PROMPT = """你是「{character_name}」，{character_role}。

## 角色设定
{system_prompt}

## 你的内心独白（玩家看不到）
{inner_monologue}

## 当前环境
{environment_desc}

## 系统标识（调用工具时必须使用）
- 你的角色 ID: `{character_id}`
- 玩家 ID: `{player_id}`

---

请根据你的内心独白和角色设定，回复玩家。

要求：
1. **台词**必须完全符合你的性格，体现你的说话风格
2. 如果需要修改游戏状态（如好感度变化、玩家位置移动、道具增减），请调用对应的工具。调用工具时，source_id 必须使用 `{character_id}`，target_id 必须使用 `{player_id}`
3. 只输出角色台词，不要加旁白或动作描写的括号注释
4. 保持台词简洁生动，一般不超过 100 字"""


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

        system_prompt = ACTION_SYSTEM_PROMPT.format(
            character_name=character.name,
            character_role=character.role,
            character_id=character.id,
            player_id=context.game_state.player.player_id,
            system_prompt=character.system_prompt,
            inner_monologue=inner_monologue,
            environment_desc=context.environment_desc,
        )

        # 构建消息列表：系统提示 + 工作记忆（历史对话） + 当前输入
        messages = [{"role": "system", "content": system_prompt}]

        # 加入工作记忆中的历史对话
        for turn in context.memory_result.working_memories[-6:]:
            if turn.role == "player":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "npc":
                messages.append({"role": "assistant", "content": turn.content})

        # 当前玩家输入
        messages.append({"role": "user", "content": context.player_input})

        # 获取工具定义
        tools = get_tool_definitions()

        logger.debug(f"[ActionGenerator] 第一次调用 LLM... tools={len(tools)} 个")

        result = self._model.complete(
            ModelRequest(
                purpose="respond",
                messages=messages,
                tools=tools if tools else [],
                temperature=0.8,
            )
        )

        # 解析响应
        dialogue = result.content or ""
        tool_call_results = []

        # ── 处理工具调用 ──
        if result.tool_calls:
            tool_registry = get_tool_registry()

            # 将 assistant 的 tool_calls 消息加入对话历史
            messages.append({
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": result.tool_calls,
            })

            for tc in result.tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                logger.info(f"[ActionGenerator] 工具调用: {func_name}({func_args})")

                # 执行工具
                tool_func = tool_registry.get(func_name)
                tool_result_data = None
                if tool_func:
                    try:
                        tool_result_data = tool_func(
                            game_state=context.game_state, **func_args
                        )
                        tool_call_results.append(ToolCallResult(
                            tool_name=func_name,
                            arguments=func_args,
                            result=tool_result_data,
                            success=True,
                        ))
                    except Exception as e:
                        logger.error(f"工具执行失败: {func_name} -> {e}")
                        tool_result_data = {"error": str(e)}
                        tool_call_results.append(ToolCallResult(
                            tool_name=func_name,
                            arguments=func_args,
                            success=False,
                            error=str(e),
                        ))
                else:
                    logger.warning(f"未知工具: {func_name}")
                    tool_result_data = {"error": f"未知工具: {func_name}"}

                # 将工具执行结果加入消息历史（反馈给 LLM）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result_data, ensure_ascii=False),
                })

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
