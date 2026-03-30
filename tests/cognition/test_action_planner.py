from __future__ import annotations

import json
from types import ModuleType
from unittest.mock import Mock

import sys

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

from src.adapters.llm.base import ModelRequest, ModelResponse
from src.cognition.action_planner import ActionPlanner
from src.cognition.perception import PerceptionContext
from src.models.character import CharacterProfile, Personality
from src.models.game_state import GameState
from src.models.memory import ConversationTurn, MemoryItem, MemoryQueryResult, MemoryType


class FakeModelAdapter:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content="别慌，先站到我身后。",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "update_affection",
                        "arguments": json.dumps(
                            {
                                "source_id": "npc_1",
                                "target_id": "player-42",
                                "delta": 10,
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
                ,
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "add_item",
                        "arguments": json.dumps(
                            {
                                "source_id": "npc_1",
                                "target_id": "player-42",
                                "item_id": "healing_potion",
                                "count": 1,
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
            usage={},
            model_name="fake-model",
        )


def make_character() -> CharacterProfile:
    return CharacterProfile(
        id="npc_1",
        name="凌霜",
        role="师姐",
        personality=Personality(
            traits=["冷静", "可靠"],
            speaking_style="简短",
        ),
        system_prompt="你是凌霜。",
    )


def make_context() -> PerceptionContext:
    game_state = GameState(session_id="session-1")
    game_state.player.player_id = "player-42"
    game_state.player.name = "行者"

    return PerceptionContext(
        player_input="救我",
        character=make_character(),
        game_state=game_state,
        memory_result=MemoryQueryResult(
            working_memories=[
                ConversationTurn(role="player", speaker_name="行者", content="救我")
            ],
            episodic_memories=[
                MemoryItem(
                    memory_type=MemoryType.EPISODIC,
                    content="曾经帮助过玩家",
                    summary="旧事",
                )
            ],
            semantic_facts=["player 与 npc 的关系为「友人」"],
            graph_relations=[],
        ),
        environment_desc="当前地点：山门",
    )


def test_action_planner_returns_tool_proposal_without_executing_tools():
    model = FakeModelAdapter()
    context = make_context()

    before_state = context.game_state.model_dump()
    plan = ActionPlanner(model_adapter=model).plan(context, "他看起来很急，我先稳住他。")
    after_state = context.game_state.model_dump()

    assert plan.dialogue == "别慌，先站到我身后。"
    assert plan.tool_name == "update_affection"
    assert plan.arguments == {
        "source_id": "npc_1",
        "target_id": "player-42",
        "delta": 10,
    }
    assert [request.purpose for request in model.requests] == ["respond"]
    assert before_state == after_state
    assert context.game_state.relationships == {}
    assert context.game_state.player.inventory == []


def test_action_planner_preserves_multiple_tool_calls_in_order():
    model = FakeModelAdapter()
    context = make_context()
    plan = ActionPlanner(model_adapter=model).plan(context, "他看起来很急，我先稳住他。")

    assert plan.dialogue == "别慌，先站到我身后。"
    assert plan.tool_name == "update_affection"
    assert plan.arguments == {
        "source_id": "npc_1",
        "target_id": "player-42",
        "delta": 10,
    }
    assert len(plan.tool_calls) == 2
    assert plan.tool_calls[0]["function"]["name"] == "update_affection"
    assert plan.tool_calls[1]["function"]["name"] == "add_item"
