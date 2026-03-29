from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

from src.runtime.actions import RuntimeAction


def test_agent_runtime_executes_propose_tool_action():
    from src.runtime.agent_runtime import AgentRuntime
    from src.runtime.turn_context import TurnContext

    class StubPolicy:
        def __init__(self):
            self.calls = 0

        def next_action(self, state):
            self.calls += 1
            if self.calls == 1:
                return RuntimeAction(
                    action_type="propose_tool",
                    payload={
                        "tool_name": "runtime_echo_tool",
                        "arguments": {"message": "hello"},
                    },
                )
            return RuntimeAction(
                action_type="respond",
                payload={"dialogue": "工具已经执行完了。"},
            )

    class StubToolExecutor:
        def __init__(self):
            self.calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

        def execute(self, tool_name, arguments=None, state=None):
            payload = dict(arguments or {})
            self.calls.append((tool_name, payload, dict(state or {})))
            return type(
                "StubExecution",
                (),
                {
                    "success": True,
                    "as_event_payload": lambda self: {
                        "tool_name": tool_name,
                        "arguments": payload,
                        "success": True,
                        "output": {"echo": payload["message"]},
                        "error": None,
                    },
                },
            )()

    tool_executor = StubToolExecutor()
    runtime = AgentRuntime(policy=StubPolicy(), tool_executor=tool_executor)
    result = runtime.run(
        TurnContext(
            turn_id="turn-tool-1",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="帮我测试工具",
            max_steps=3,
        )
    )

    assert result.dialogue == "工具已经执行完了。"
    assert result.stop_reason == "response_generated"
    assert tool_executor.calls[0][0] == "runtime_echo_tool"
    assert tool_executor.calls[0][1] == {"message": "hello"}
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
    assert result.trace.events[3].payload["tool_name"] == "runtime_echo_tool"
    assert result.trace.events[3].payload["success"] is True
    assert result.trace.events[3].payload["output"] == {"echo": "hello"}


def test_agent_runtime_default_tool_path_exposes_execution_to_next_action():
    from src.runtime.agent_runtime import AgentRuntime
    from src.runtime.turn_context import TurnContext

    class StubPolicy:
        def __init__(self):
            self.calls = 0

        def next_action(self, state):
            self.calls += 1
            if self.calls == 1:
                return RuntimeAction(
                    action_type="propose_tool",
                    payload={
                        "tool_name": "add_item",
                        "arguments": {"item_name": "玉佩"},
                    },
                )

            execution = state["last_tool_execution"]
            return RuntimeAction(
                action_type="respond",
                payload={"dialogue": f'{execution.output["item_name"]}已加入背包。'},
            )

    runtime = AgentRuntime(policy=StubPolicy())
    result = runtime.run(
        TurnContext(
            turn_id="turn-tool-2",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="给我一个道具",
            max_steps=3,
        )
    )

    assert result.dialogue == "玉佩已加入背包。"
    assert result.stop_reason == "response_generated"
    assert result.trace.events[3].payload["tool_name"] == "add_item"
    assert result.trace.events[3].payload["success"] is True
    assert result.trace.events[3].payload["output"] == {
        "item_name": "玉佩",
        "action": "added",
    }
