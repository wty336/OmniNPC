"""
API 端点测试。
"""

import pytest
from unittest.mock import patch, Mock

from fastapi.testclient import TestClient

from src.models.message import AgentResponse


@pytest.fixture
def client():
    """创建测试用 FastAPI 客户端。"""
    from src.api.app import app
    return TestClient(app)


def test_health_check(client):
    """测试健康检查端点。"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "OmniNPC"


def test_chat_missing_fields(client):
    """测试缺少必填字段时的错误处理。"""
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422  # Validation Error


def test_chat_endpoint_format(client):
    """测试对话端点的请求格式。"""
    # 模拟引擎返回
    mock_response = AgentResponse(
        dialogue="哼，什么事？",
        inner_monologue="这个笨蛋又来了...",
        character_id="tsundere_sister",
        character_name="凌霜",
    )

    with patch("src.api.routes.chat.get_engine") as mock_engine:
        engine_instance = mock_engine.return_value
        engine_instance.process_chat = Mock(return_value=mock_response)

        response = client.post("/api/v1/chat", json={
            "player_input": "你好",
            "character_id": "tsundere_sister",
            "session_id": "test",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["dialogue"] == "哼，什么事？"
