from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import Mock, patch

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

from src.adapters.llm.base import ModelRequest, ModelResponse
from src.cognition.action_generator import ActionGenerator
from src.cognition.inner_monologue import InnerMonologue
from src.cognition.perception import PerceptionContext
from src.models.character import CharacterProfile, Personality
from src.models.game_state import GameState
from src.models.memory import ConversationTurn, MemoryItem, MemoryQueryResult, MemoryType


class FakeModelAdapter:
    def __init__(self, response_tool_calls: list[dict] | None = None) -> None:
        self.requests: list[ModelRequest] = []
        self.response_tool_calls = response_tool_calls or []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)

        if request.purpose == "reflect":
            return ModelResponse(
                content="他看起来很急，我先稳住他。",
                tool_calls=[],
                usage={},
                model_name="fake-model",
            )

        respond_calls = sum(1 for item in self.requests if item.purpose == "respond")
        if respond_calls == 1:
            return ModelResponse(
                content="",
                tool_calls=self.response_tool_calls,
                usage={},
                model_name="fake-model",
            )

        return ModelResponse(
            content="别慌，先站到我身后。",
            tool_calls=[],
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


def make_tool_call(name: str, arguments: dict[str, object], call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def test_reflection_and_action_use_injected_model_adapter_and_preserve_tools():
    model = FakeModelAdapter(
        response_tool_calls=[
            make_tool_call(
                "update_affection",
                {
                    "source_id": "npc_1",
                    "target_id": "player-42",
                    "delta": 10,
                },
                "call-1",
            )
        ]
    )
    context = make_context()

    def update_affection(*, game_state: GameState, source_id: str, target_id: str, delta: float):
        new_value = game_state.update_affection(source_id, target_id, delta)
        return {"new_affection": new_value}

    with patch("src.cognition.action_generator.get_tool_definitions", return_value=[{"type": "function"}]), patch(
        "src.cognition.action_generator.get_tool_registry",
        return_value={"update_affection": update_affection},
    ):
        monologue = InnerMonologue(model_adapter=model).generate(context)
        response = ActionGenerator(model_adapter=model).generate(context, monologue)

    assert monologue == "他看起来很急，我先稳住他。"
    assert response.dialogue == "别慌，先站到我身后。"
    assert [request.purpose for request in model.requests] == ["reflect", "respond", "respond"]
    assert response.tool_calls[0].tool_name == "update_affection"
    assert response.tool_calls[0].success is True
    assert context.game_state.get_relationship("npc_1", "player-42").affection == 60


def test_action_generator_executes_multiple_tool_calls_and_replays_all_results():
    model = FakeModelAdapter(
        response_tool_calls=[
            make_tool_call(
                "update_affection",
                {
                    "source_id": "npc_1",
                    "target_id": "player-42",
                    "delta": 10,
                },
                "call-1",
            ),
            make_tool_call(
                "add_item",
                {
                    "source_id": "npc_1",
                    "target_id": "player-42",
                    "item_id": "healing_potion",
                    "count": 1,
                },
                "call-2",
            ),
        ]
    )
    context = make_context()

    def update_affection(*, game_state: GameState, source_id: str, target_id: str, delta: float):
        new_value = game_state.update_affection(source_id, target_id, delta)
        return {"new_affection": new_value}

    def add_item(*, game_state: GameState, source_id: str, target_id: str, item_id: str, count: int):
        for _ in range(count):
            game_state.player.inventory.append(item_id)
        return {"item_id": item_id, "count": count}

    with patch("src.cognition.action_generator.get_tool_definitions", return_value=[{"type": "function"}]), patch(
        "src.cognition.action_generator.get_tool_registry",
        return_value={
            "update_affection": update_affection,
            "add_item": add_item,
        },
    ):
        monologue = InnerMonologue(model_adapter=model).generate(context)
        response = ActionGenerator(model_adapter=model).generate(context, monologue)

    assert response.dialogue == "别慌，先站到我身后。"
    assert [tool_call.tool_name for tool_call in response.tool_calls] == ["update_affection", "add_item"]
    assert [request.purpose for request in model.requests] == ["reflect", "respond", "respond"]

    second_request = model.requests[-1]
    assistant_message = second_request.messages[-3]
    tool_message_1 = second_request.messages[-2]
    tool_message_2 = second_request.messages[-1]

    assert assistant_message["tool_calls"][0]["function"]["name"] == "update_affection"
    assert assistant_message["tool_calls"][1]["function"]["name"] == "add_item"
    assert tool_message_1["tool_call_id"] == "call-1"
    assert tool_message_2["tool_call_id"] == "call-2"
    assert context.game_state.get_relationship("npc_1", "player-42").affection == 60
    assert context.game_state.player.inventory.count("healing_potion") == 1


def test_reflection_and_action_default_constructors_use_internal_model_adapter():
    fake_model = FakeModelAdapter(
        response_tool_calls=[
            make_tool_call(
                "update_affection",
                {
                    "source_id": "npc_1",
                    "target_id": "player-42",
                    "delta": 10,
                },
                "call-1",
            )
        ]
    )
    context = make_context()

    def update_affection(*, game_state: GameState, source_id: str, target_id: str, delta: float):
        new_value = game_state.update_affection(source_id, target_id, delta)
        return {"new_affection": new_value}

    with patch("src.cognition.inner_monologue.ArkModelAdapter", return_value=fake_model), patch(
        "src.cognition.action_generator.ArkModelAdapter",
        return_value=fake_model,
    ), patch("src.cognition.action_generator.get_tool_definitions", return_value=[{"type": "function"}]), patch(
        "src.cognition.action_generator.get_tool_registry",
        return_value={"update_affection": update_affection},
    ):
        monologue = InnerMonologue().generate(context)
        response = ActionGenerator().generate(context, monologue)

    assert monologue == "他看起来很急，我先稳住他。"
    assert response.dialogue == "别慌，先站到我身后。"
