from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    model_name: str

