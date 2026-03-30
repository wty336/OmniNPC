from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.runtime.actions import RuntimeAction


class ChatRuntimePolicy:
    def next_action(self, state: dict[str, Any]) -> RuntimeAction:
        if state.get("memory_snapshot") is None:
            return RuntimeAction(action_type="retrieve_memory")

        if state.get("reflection") is None:
            return RuntimeAction(action_type="reflect")

        if state.get("planned_response") is None:
            return RuntimeAction(action_type="plan_response")

        planned_response = state["planned_response"]
        tool_name = self._read_field(planned_response, "tool_name", "")

        if tool_name and state.get("last_tool_execution") is None:
            return RuntimeAction(
                action_type="propose_tool",
                payload={
                    "tool_name": tool_name,
                    "arguments": self._read_field(planned_response, "arguments", {}) or {},
                },
            )

        return RuntimeAction(
            action_type="respond",
            payload={"dialogue": self._read_field(planned_response, "dialogue", "")},
        )

    @staticmethod
    def _read_field(value: Any, name: str, default: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)
