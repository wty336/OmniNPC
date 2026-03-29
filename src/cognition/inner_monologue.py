"""
内心独白模块 (Inner Monologue) — 认知管线的第二步。

职责：
1. 根据 NPC 人设 + 记忆上下文，生成一段后台心理活动
2. 内心独白仅后台可见，不展示给玩家
3. 内心独白将作为后续台词生成的重要依据，确保 NPC 动机一致
"""

from __future__ import annotations

from loguru import logger

from src.adapters.llm.ark_adapter import ArkModelAdapter
from src.adapters.llm.base import ModelRequest
from src.cognition.perception import PerceptionContext

# 内心独白生成的 Prompt 模板
INNER_MONOLOGUE_PROMPT = """你是「{character_name}」，{character_role}。

## 你的性格
{personality_desc}

## 当前环境
{environment_desc}

## 你记得的相关事件
{episodic_context}

## 你了解的关系
{semantic_context}

## 最近的对话
{working_context}

---

现在，玩家对你说了这句话：
「{player_input}」

请以第一人称写出你的**内心独白**（不超过 150 字）。
要求：
1. 体现你的性格特征和说话风格
2. 分析玩家的意图和你对此的真实感受
3. 思考你接下来应该如何回应（是帮忙、拒绝、试探还是别的？）
4. 只输出内心独白文本，不要加任何前缀或标注"""


class InnerMonologue:
    """
    内心独白生成器。

    强制 NPC 在回复前先产生一段后台心理活动，
    有效解决 NPC 动机不明和"讨好型人格"问题。
    """

    def __init__(self, model_adapter: ArkModelAdapter | None = None):
        self._model = model_adapter or ArkModelAdapter()

    def generate(self, context: PerceptionContext) -> str:
        """
        生成内心独白。

        Parameters
        ----------
        context : 感知阶段输出的完整上下文

        Returns
        -------
        str : 内心独白文本
        """
        character = context.character
        memory = context.memory_result

        # 构建 episodic 上下文
        episodic_lines = []
        for mem in memory.episodic_memories[:3]:
            episodic_lines.append(f"- {mem.content[:100]}")
        episodic_context = "\n".join(episodic_lines) if episodic_lines else "（没有特别相关的记忆）"

        # 构建 semantic 上下文
        semantic_context = "\n".join(
            [f"- {fact}" for fact in memory.semantic_facts[:5]]
        ) if memory.semantic_facts else "（暂无关系信息）"

        # 构建 working 上下文（最近对话）
        working_lines = []
        for turn in memory.working_memories[-5:]:
            speaker = turn.speaker_name or turn.role
            working_lines.append(f"{speaker}: {turn.content}")
        working_context = "\n".join(working_lines) if working_lines else "（这是你们的第一次交流）"

        # 构建 personality 描述
        personality_desc = (
            f"性格特征：{', '.join(character.personality.traits)}\n"
            f"说话风格：{character.personality.speaking_style}"
        )

        prompt = INNER_MONOLOGUE_PROMPT.format(
            character_name=character.name,
            character_role=character.role,
            personality_desc=personality_desc,
            environment_desc=context.environment_desc,
            episodic_context=episodic_context,
            semantic_context=semantic_context,
            working_context=working_context,
            player_input=context.player_input,
        )

        logger.debug(f"[InnerMonologue] 生成中... (character={character.name})")

        result = self._model.complete(
            ModelRequest(
                purpose="reflect",
                messages=[
                    {"role": "system", "content": "你是一个角色扮演内心独白生成器。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=300,
            )
        )

        monologue = result.content.strip()
        logger.info(f"[InnerMonologue] 「{character.name}」内心OS: {monologue[:80]}...")
        return monologue
