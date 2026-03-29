from src.runtime.actions import RuntimeAction
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.turn_context import TurnContext


class TracePolicy:
    def __init__(self):
        self._actions = [
            RuntimeAction(action_type="retrieve_memory"),
            RuntimeAction(action_type="respond", payload={"dialogue": "我来处理。"}),
        ]

    def next_action(self, state):
        return self._actions.pop(0)


def test_trace_contains_expected_event_order_for_multi_step_turn():
    runtime = AgentRuntime(policy=TracePolicy())
    result = runtime.run(
        TurnContext(
            turn_id="turn-4",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="师姐，快帮我。",
            max_steps=3,
        )
    )

    event_types = [event.event_type.value for event in result.trace.events]

    assert event_types == [
        "turn_started",
        "step_started",
        "action_selected",
        "step_completed",
        "step_started",
        "action_selected",
        "step_completed",
        "turn_finished",
    ]
    assert result.trace.events[2].payload["action_type"] == "retrieve_memory"
    assert result.trace.events[5].payload["action_type"] == "respond"
    assert result.trace.events[-1].payload["stop_reason"] == "response_generated"
    assert result.trace.events[-1].payload["dialogue"] == "我来处理。"
