from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.executor import ToolExecutor
from src.models.game_state import GameState


def test_tool_catalog_supports_direct_registration():
    def echo_tool(value: str) -> dict[str, str]:
        return {"echo": value}

    catalog = ToolCatalog()
    catalog.register(
        name="echo_tool",
        description="Echo a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        func=echo_tool,
    )

    tool = catalog.get("echo_tool")

    assert tool is not None
    assert tool.name == "echo_tool"
    assert catalog.as_openai_tools() == [
        {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "Echo a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]


def test_tool_catalog_exposes_legacy_tool_registration():
    catalog = ToolCatalog.load()

    tool_names = {tool.name for tool in catalog.tools}

    assert "add_item" in tool_names
    assert "remove_item" in tool_names
    assert "update_affection" in tool_names
    assert "update_player_location" in tool_names
    assert "set_world_flag" in tool_names


def test_tool_executor_runs_registered_tool_and_injects_game_state():
    catalog = ToolCatalog.load()
    executor = ToolExecutor(catalog)
    game_state = GameState(session_id="session-1")

    result = executor.execute(
        "add_item",
        {"item_name": "玉佩"},
        {"game_state": game_state},
    )

    assert result.tool_name == "add_item"
    assert result.success is True
    assert result.output == {"item_name": "玉佩", "action": "added"}
    assert game_state.player.inventory == ["玉佩"]


def test_tool_executor_returns_failure_for_unknown_tool():
    catalog = ToolCatalog.load()
    executor = ToolExecutor(catalog)

    result = executor.execute("missing_tool", {})

    assert result.tool_name == "missing_tool"
    assert result.success is False
    assert "missing_tool" in result.error
