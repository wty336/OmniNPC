from __future__ import annotations

from src.adapters.llm.base import ModelRequest, ModelResponse


def get_llm_client():
    from src.llm.client import get_llm_client as _get_llm_client

    return _get_llm_client()


class ArkModelAdapter:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raw = get_llm_client().chat(
            messages=request.messages,
            tools=request.tools or None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ModelResponse(
            content=raw.get("content", ""),
            tool_calls=raw.get("tool_calls") or [],
            usage=raw.get("usage", {}),
            model_name=raw.get("model_name", "ark"),
        )
