from __future__ import annotations

from typing import Any, Protocol

from src.runtime.actions import RuntimeAction
from src.runtime.chat_policy import ChatRuntimePolicy


class RuntimePolicy(Protocol):
    def next_action(self, state: dict[str, Any]) -> RuntimeAction:
        ...


class DefaultRuntimePolicy:
    def next_action(self, state: dict[str, Any]) -> RuntimeAction:
        return RuntimeAction(action_type="respond", payload={"dialogue": "我在。"})
