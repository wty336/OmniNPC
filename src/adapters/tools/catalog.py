from __future__ import annotations

import importlib
from typing import Any

from src.adapters.tools.types import ToolDefinition
from src.tools.base import get_tool_definitions, get_tool_registry

_LEGACY_TOOL_MODULES = (
    "src.tools.state_updater",
    "src.tools.item_manager",
)


def _ensure_legacy_tool_modules_loaded() -> None:
    for module_name in _LEGACY_TOOL_MODULES:
        importlib.import_module(module_name)


def _tool_metadata_by_name() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for entry in get_tool_definitions():
        function = entry.get("function", {})
        name = function.get("name")
        if not name:
            continue
        metadata[name] = {
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }
    return metadata


class ToolCatalog:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Any,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=func,
        )

    @classmethod
    def load(cls) -> "ToolCatalog":
        _ensure_legacy_tool_modules_loaded()

        registry = get_tool_registry()
        metadata = _tool_metadata_by_name()
        catalog = cls()
        for name, function in registry.items():
            catalog.register(
                name=name,
                description=metadata.get(name, {}).get("description", ""),
                parameters=metadata.get(name, {}).get("parameters", {}),
                func=function,
            )
        return catalog

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self.tools]
