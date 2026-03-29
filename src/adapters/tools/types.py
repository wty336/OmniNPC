from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    function: Callable[..., Any] = Field(repr=False)

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    output: Any = None
    error: str | None = None

    def as_event_payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }
