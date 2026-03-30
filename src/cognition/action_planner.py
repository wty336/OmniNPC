"""行动规划模块 (Action Planner)。

职责：
1. 结合感知上下文 + 内心独白，生成对外台词候选
2. 仅提议单个工具调用，不执行任何工具
3. 供 ActionGenerator 与后续 runtime-enabled 路径复用
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.adapters.llm.ark_adapter import ArkModelAdapter
from src.adapters.llm.base import ModelRequest
from src.cognition.perception import PerceptionContext
from src.tools.base import get_tool_definitions

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


@dataclass(slots=True)
class ActionPlan:
    dialogue: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def build_action_messages(
    context: PerceptionContext,
    inner_monologue: str,
) -> list[dict[str, Any]]:
    """构建行动规划/执行共用的消息列表。"""
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

    messages = [{"role": "system", "content": system_prompt}]

    for turn in context.memory_result.working_memories[-6:]:
        if turn.role == "player":
            messages.append({"role": "user", "content": turn.content})
        elif turn.role == "npc":
            messages.append({"role": "assistant", "content": turn.content})

    messages.append({"role": "user", "content": context.player_input})
    return messages


class ActionPlanner:
    """无副作用的行动规划器。"""

    def __init__(self, model_adapter: ArkModelAdapter | None = None):
        self._model = model_adapter or ArkModelAdapter()

    def plan(
        self,
        context: PerceptionContext,
        inner_monologue: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> ActionPlan:
        """生成对话计划与单个工具提议，不执行工具。"""
        messages = build_action_messages(context, inner_monologue)
        tool_definitions = tools if tools is not None else get_tool_definitions()

        logger.debug(f"[ActionPlanner] 调用 LLM 进行规划... tools={len(tool_definitions)} 个")
        result = self._model.complete(
            ModelRequest(
                purpose="respond",
                messages=messages,
                tools=tool_definitions if tool_definitions else [],
                temperature=0.8,
            )
        )

        plan = ActionPlan(dialogue=result.content or "")

        if result.tool_calls:
            plan.tool_calls = list(result.tool_calls)
            first_tool_call = plan.tool_calls[0]
            function = first_tool_call.get("function", {})
            plan.tool_name = function.get("name")
            raw_arguments = function.get("arguments", "{}")
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
            if isinstance(parsed_arguments, dict):
                plan.arguments = parsed_arguments
            else:
                plan.arguments = {}

        return plan
