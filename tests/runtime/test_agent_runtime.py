import importlib
import sys

import pytest

from src.runtime.actions import RuntimeAction


def _clear_runtime_modules() -> None:
    for module_name in (
        "src.runtime",
        "src.runtime.actions",
        "src.runtime.agent_runtime",
        "src.runtime.policies",
        "src.runtime.stop_conditions",
    ):
        sys.modules.pop(module_name, None)


def test_agent_runtime_propagates_default_dialogue():
    _clear_runtime_modules()

    agent_runtime_module = importlib.import_module("src.runtime.agent_runtime")
    policies_module = importlib.import_module("src.runtime.policies")
    turn_context_module = importlib.import_module("src.runtime.turn_context")

    runtime = agent_runtime_module.AgentRuntime(
        policy=policies_module.DefaultRuntimePolicy()
    )
    result = runtime.run(
        turn_context_module.TurnContext(
            turn_id="turn-2",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="师姐救我",
        )
    )

    assert result.dialogue == "我在。"
    assert result.stop_reason == "response_generated"
    assert [event.event_type.value for event in result.trace.events] == [
        "turn_started",
        "step_started",
        "action_selected",
        "step_completed",
        "turn_finished",
    ]


def test_agent_runtime_exhausts_budget_without_response():
    _clear_runtime_modules()

    actions_module = importlib.import_module("src.runtime.actions")
    agent_runtime_module = importlib.import_module("src.runtime.agent_runtime")
    turn_context_module = importlib.import_module("src.runtime.turn_context")

    class StubPolicy:
        def next_action(self, state):
            return actions_module.RuntimeAction(action_type="inspect")

    runtime = agent_runtime_module.AgentRuntime(policy=StubPolicy())
    result = runtime.run(
        turn_context_module.TurnContext(
            turn_id="turn-3",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="继续想",
            max_steps=2,
        )
    )

    assert result.dialogue == ""
    assert result.stop_reason == "step_budget_exhausted"
    assert [event.event_type.value for event in result.trace.events] == [
        "turn_started",
        "step_started",
        "action_selected",
        "step_completed",
        "step_started",
        "action_selected",
        "step_completed",
        "turn_finished",
    ]


def test_runtime_action_rejects_empty_respond_dialogue():
    with pytest.raises(ValueError):
        RuntimeAction(action_type="respond")
