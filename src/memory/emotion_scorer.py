"""
情绪打分器 (Emotion Scorer) — 基于 LLM 评估对话的情绪强度和重要性。

每次对话结束后，调用 LLM 对本次交互进行评估，
输出 emotion_score 和 importance 两个指标，
作为遗忘曲线的关键参数。
"""

from __future__ import annotations

from loguru import logger

from src.llm.client import get_llm_client

# 情绪打分 Prompt
EMOTION_SCORING_PROMPT = """你是一个对话情感分析器。请分析以下 NPC 与玩家的对话交互，评估该交互对 NPC 的情绪影响和事件重要性。

## NPC 信息
- 角色: {character_name}（{character_role}）
- 与玩家当前好感度: {affection}

## 本次对话
玩家说: 「{player_input}」
{character_name}回复: 「{npc_response}」
{character_name}的内心独白: 「{inner_monologue}」

---

请以 JSON 格式输出你的评估（不要输出其他内容）:
{{
    "emotion_score": <float 0-10, 该交互引发的情绪强度。0=毫无波动，10=极度强烈>,
    "importance": <float 0-10, 对整体剧情的重要性。0=无关紧要，10=关键转折>,
    "emotion_type": "<情绪类型: 开心/感动/生气/伤心/害羞/恐惧/惊讶/厌恶/平淡>",
    "reason": "<简短解释>"
}}"""


class EmotionScorer:
    """
    情绪打分器。

    调用 LLM 对每次交互进行情感评估，
    返回 emotion_score 和 importance 作为遗忘曲线参数。
    """

    def score(
        self,
        player_input: str,
        npc_response: str,
        inner_monologue: str,
        character_name: str,
        character_role: str,
        affection: float = 50.0,
    ) -> dict:
        """
        评估一次交互的情绪强度和重要性。

        Returns
        -------
        dict : {"emotion_score", "importance", "emotion_type", "reason"}
        """
        prompt = EMOTION_SCORING_PROMPT.format(
            character_name=character_name,
            character_role=character_role,
            affection=affection,
            player_input=player_input,
            npc_response=npc_response,
            inner_monologue=inner_monologue or "（无）",
        )

        llm = get_llm_client()
        try:
            result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "你是一个精确的情感分析 AI，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            emotion_score = max(0.0, min(10.0, float(result.get("emotion_score", 5.0))))
            importance = max(0.0, min(10.0, float(result.get("importance", 5.0))))
            emotion_type = result.get("emotion_type", "平淡")
            reason = result.get("reason", "")

            logger.info(
                f"[EmotionScorer] 评分: emotion={emotion_score:.1f}, "
                f"importance={importance:.1f}, type={emotion_type} ({reason})"
            )

            return {
                "emotion_score": emotion_score,
                "importance": importance,
                "emotion_type": emotion_type,
                "reason": reason,
            }

        except Exception as e:
            logger.warning(f"[EmotionScorer] 评分失败，使用默认值: {e}")
            return {
                "emotion_score": 5.0,
                "importance": 5.0,
                "emotion_type": "平淡",
                "reason": "评分失败，使用默认值",
            }
