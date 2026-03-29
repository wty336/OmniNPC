import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

loguru_stub = ModuleType("loguru")
loguru_stub.logger = Mock()
sys.modules.setdefault("loguru", loguru_stub)

config_stub = ModuleType("config")
config_settings_stub = ModuleType("config.settings")
config_settings_stub.settings = SimpleNamespace(
    llm=SimpleNamespace(
        api_key="test-key",
        model_endpoint="ark-test-endpoint",
        temperature=0.75,
        max_tokens=256,
        top_p=0.9,
    )
)
config_stub.settings = config_settings_stub
sys.modules.setdefault("config", config_stub)
sys.modules.setdefault("config.settings", config_settings_stub)

from src.adapters.llm.ark_adapter import ArkModelAdapter
from src.adapters.llm.base import ModelRequest, ModelResponse
from src.llm.client import LLMClient
from src.observability.events import EventType


def test_model_request_and_response_reject_unexpected_fields():
    with pytest.raises(ValidationError):
        ModelRequest(
            purpose="respond",
            messages=[{"role": "user", "content": "你好"}],
            tools=[],
            temperature=0.5,
            max_tokens=64,
            unexpected="value",
        )

    with pytest.raises(ValidationError):
        ModelResponse(
            content="收到",
            tool_calls=[],
            usage={},
            model_name="fake-model",
            unexpected="value",
        )


def test_ark_model_adapter_translates_chat_response():
    fake_client = Mock()
    fake_client.chat.return_value = {
        "content": "收到。",
        "tool_calls": [{"id": "call-1"}],
        "usage": {"total_tokens": 10},
        "model_name": "ark-test-endpoint",
    }

    request = ModelRequest(
        purpose="respond",
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.7,
        max_tokens=128,
    )

    with patch("src.adapters.llm.ark_adapter.get_llm_client", return_value=fake_client):
        response = ArkModelAdapter().complete(request)

    fake_client.chat.assert_called_once_with(
        messages=[{"role": "user", "content": "你好"}],
        tools=None,
        temperature=0.7,
        max_tokens=128,
    )
    assert response == ModelResponse(
        content="收到。",
        tool_calls=[{"id": "call-1"}],
        usage={"total_tokens": 10},
        model_name="ark-test-endpoint",
    )


def test_ark_model_adapter_preserves_zero_temperature_override():
    fake_client = Mock()
    fake_client.chat.return_value = {
        "content": "收到。",
        "tool_calls": [],
        "usage": {"total_tokens": 1},
        "model_name": "ark-test-endpoint",
    }

    request = ModelRequest(
        purpose="respond",
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.0,
        max_tokens=128,
    )

    with patch("src.adapters.llm.ark_adapter.get_llm_client", return_value=fake_client):
        ArkModelAdapter().complete(request)

    fake_client.chat.assert_called_once_with(
        messages=[{"role": "user", "content": "你好"}],
        tools=None,
        temperature=0.0,
        max_tokens=128,
    )


def test_llm_client_preserves_explicit_zero_values():
    fake_transport = Mock()
    fake_transport.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="收到。",
                    role="assistant",
                    tool_calls=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    client = object.__new__(LLMClient)
    client._client = fake_transport
    client._model = "ark-test-endpoint"

    with patch(
        "src.llm.client.settings",
        new=SimpleNamespace(
            llm=SimpleNamespace(
                temperature=0.75,
                max_tokens=256,
                top_p=0.9,
            )
        ),
    ):
        result = client.chat(
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.0,
            max_tokens=0,
        )

    fake_transport.chat.completions.create.assert_called_once_with(
        model="ark-test-endpoint",
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.0,
        max_tokens=0,
        top_p=0.9,
    )
    assert result["model_name"] == "ark-test-endpoint"


def test_event_type_includes_model_called():
    assert EventType.MODEL_CALLED.value == "model_called"
