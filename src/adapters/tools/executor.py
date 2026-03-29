from __future__ import annotations

import inspect
from typing import Any

from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.types import ToolExecutionResult


class ToolExecutor:
    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        tool = self.catalog.get(tool_name)
        if tool is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=dict(arguments or {}),
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        call_arguments = dict(arguments or {})
        signature = inspect.signature(tool.function)

        if state:
            if "game_state" in signature.parameters and "game_state" not in call_arguments:
                call_arguments["game_state"] = state.get("game_state")
            if "character_id" in signature.parameters and "character_id" not in call_arguments:
                call_arguments["character_id"] = state.get("character_id")

        try:
            result = tool.function(**call_arguments)
        except Exception as exc:  # pragma: no cover - defensive path
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=dict(arguments or {}),
                success=False,
                error=str(exc),
            )

        return ToolExecutionResult(
            tool_name=tool_name,
            arguments=dict(arguments or {}),
            success=True,
            output=result,
        )
