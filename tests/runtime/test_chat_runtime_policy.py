from types import SimpleNamespace

import pytest

from src.models.game_state import GameState
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.chat_policy import ChatRuntimePolicy
from src.runtime.policies import ChatRuntimePolicy as ExportedChatRuntimePolicy
from src.runtime.turn_context import TurnContext


class FakeMemoryResult:
    def model_dump(self):
        return {
            "working_memories": [],
            "episodic_memories": [],
            "semantic_facts": [],
            "graph_relations": [],
        }


class FakeMemoryAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, query: str, player_id: str = "player_default"):
        self.calls.append((query, player_id))
        return FakeMemoryResult()


class FakeReflector:
    def __init__(self):
        self.calls: list[object] = []

    def generate(self, context):
        self.calls.append(context)
        return "先观察"


class FakeActionPlanner:
    def __init__(self):
        self.calls: list[tuple[object, str]] = []
        self.finalize_calls: list[tuple[object, str, object, list[object]]] = []

    def plan(self, context, inner_monologue: str):
        self.calls.append((context, inner_monologue))
        return SimpleNamespace(
            dialogue="先帮你看看。",
            tool_name="inspect_item",
            arguments={"item": "玉佩"},
        )

    def finalize_response(self, context, inner_monologue: str, plan, tool_results):
        self.finalize_calls.append((context, inner_monologue, plan, list(tool_results)))
        return getattr(plan, "dialogue", "")


class FakeToolExecution:
    def __init__(self, tool_name: str, arguments: dict[str, object]):
        self.tool_name = tool_name
        self.arguments = arguments
        self.success = True
        self.output = {"tool_name": tool_name, "arguments": arguments}

    def as_event_payload(self):
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "success": True,
            "output": self.output,
            "error": None,
        }


class FakeToolExecutor:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def execute(self, tool_name, arguments=None, state=None):
        payload = dict(arguments or {})
        self.calls.append((tool_name, payload, dict(state or {})))
        return FakeToolExecution(tool_name, payload)


class EmptyDialogueToolPlanner:
    def __init__(self):
        self.calls: list[tuple[object, str]] = []
        self.finalize_calls: list[tuple[object, str, object, list[object]]] = []

    def plan(self, context, inner_monologue: str):
        self.calls.append((context, inner_monologue))
        return SimpleNamespace(
            dialogue="",
            tool_name="inspect_item",
            arguments={"item": "玉佩"},
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "inspect_item",
                        "arguments": '{"item": "玉佩"}',
                    },
                }
            ],
        )

    def finalize_response(self, context, inner_monologue: str, plan, tool_results):
        self.finalize_calls.append((context, inner_monologue, plan, list(tool_results)))
        return "别慌，东西我先替你收着。"


def test_chat_runtime_policy_is_importable_from_runtime_policies():
    assert ExportedChatRuntimePolicy is ChatRuntimePolicy


def test_chat_runtime_policy_prioritizes_memory_reflection_planning_tool_and_response():
    policy = ChatRuntimePolicy()

    state = {
        "memory_snapshot": None,
        "reflection": None,
        "planned_response": None,
        "last_tool_execution": None,
    }

    action = policy.next_action(state)
    assert action.action_type == "retrieve_memory"

    state["memory_snapshot"] = {}
    action = policy.next_action(state)
    assert action.action_type == "reflect"

    state["reflection"] = "先观察"
    action = policy.next_action(state)
    assert action.action_type == "plan_response"

    state["planned_response"] = {
        "dialogue": "先帮你看看。",
        "tool_name": "inspect_item",
        "arguments": {"item": "玉佩"},
    }
    action = policy.next_action(state)
    assert action.action_type == "propose_tool"
    assert action.payload == {
        "tool_name": "inspect_item",
        "arguments": {"item": "玉佩"},
    }

    state["last_tool_execution"] = {"success": True}
    action = policy.next_action(state)
    assert action.action_type == "respond"
    assert action.payload == {"dialogue": "先帮你看看。"}


def test_chat_runtime_runtime_preserves_intermediate_state_through_tool_response():
    memory_adapter = FakeMemoryAdapter()
    reflector = FakeReflector()
    planner = FakeActionPlanner()
    tool_executor = FakeToolExecutor()
    perception_context = SimpleNamespace(tag="perception")
    game_state = GameState(session_id="session-1")
    game_state.player.player_id = "player-9"

    runtime = AgentRuntime(
        policy=ChatRuntimePolicy(),
        tool_executor=tool_executor,
        memory_adapter=memory_adapter,
        reflector=reflector,
        action_planner=planner,
        perception_context=perception_context,
        game_state=game_state,
    )

    result = runtime.run(
        TurnContext(
            turn_id="turn-chat-1",
            session_id="session-1",
            character_id="npc-1",
            player_input="帮我看看这个东西",
            max_steps=6,
        )
    )

    assert result.dialogue == "先帮你看看。"
    assert result.stop_reason == "response_generated"
    assert memory_adapter.calls == [("帮我看看这个东西", "player-9")]
    assert reflector.calls == [perception_context]
    assert planner.calls == [(perception_context, "先观察")]
    assert tool_executor.calls[0][0] == "inspect_item"
    assert tool_executor.calls[0][1] == {"item": "玉佩"}
    assert tool_executor.calls[0][2]["memory_snapshot"] == FakeMemoryResult().model_dump()
    assert tool_executor.calls[0][2]["reflection"] == "先观察"
    assert tool_executor.calls[0][2]["planned_response"].dialogue == "先帮你看看。"
    assert [event.event_type.value for event in result.trace.events] == [
        "turn_started",
        "step_started",
        "action_selected",
        "step_completed",
        "step_started",
        "action_selected",
        "step_completed",
        "step_started",
        "action_selected",
        "step_completed",
        "step_started",
        "action_selected",
        "step_completed",
        "step_started",
        "action_selected",
        "step_completed",
        "turn_finished",
    ]


def test_chat_runtime_policy_requires_collaborators_before_running():
    runtime = AgentRuntime(policy=ChatRuntimePolicy())

    with pytest.raises(ValueError, match="memory_adapter|reflector|action_planner|perception_context"):
        runtime.run(
            TurnContext(
                turn_id="turn-misconfigured",
                session_id="session-1",
                character_id="npc-1",
                player_input="帮我看看这个东西",
                max_steps=1,
            )
        )


def test_chat_runtime_generates_final_dialogue_after_tool_execution_when_initial_plan_is_empty():
    memory_adapter = FakeMemoryAdapter()
    reflector = FakeReflector()
    planner = EmptyDialogueToolPlanner()
    tool_executor = FakeToolExecutor()
    perception_context = SimpleNamespace(tag="perception")
    game_state = GameState(session_id="session-1")
    game_state.player.player_id = "player-9"

    runtime = AgentRuntime(
        policy=ChatRuntimePolicy(),
        tool_executor=tool_executor,
        memory_adapter=memory_adapter,
        reflector=reflector,
        action_planner=planner,
        perception_context=perception_context,
        game_state=game_state,
    )

    result = runtime.run(
        TurnContext(
            turn_id="turn-chat-empty-dialogue",
            session_id="session-1",
            character_id="npc-1",
            player_input="帮我看看这个东西",
            max_steps=6,
        )
    )

    assert result.dialogue == "别慌，东西我先替你收着。"
    assert planner.finalize_calls
