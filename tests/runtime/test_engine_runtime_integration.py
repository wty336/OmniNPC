from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

from src.models.message import AgentResponse
from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.runtime_result import RuntimeResult


class _FakeGameStateStore:
    def __init__(self):
        self.saved_states = []
        self.loaded_session_ids = []

    def load_or_create(self, session_id: str):
        self.loaded_session_ids.append(session_id)
        return SimpleNamespace(
            session_id=session_id,
            player=SimpleNamespace(location="起始地点"),
            npcs={},
        )

    def save(self, state):
        self.saved_states.append(state)


def _install_loguru_stub() -> None:
    if "loguru" in sys.modules:
        return

    class _Logger:
        def info(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    stub = types.ModuleType("loguru")
    stub.logger = _Logger()
    sys.modules["loguru"] = stub


def _install_game_state_store_stub() -> None:
    stub = types.ModuleType("src.storage.game_state_store")
    stub.GameStateStore = _FakeGameStateStore
    sys.modules["src.storage.game_state_store"] = stub


def _install_settings_stub() -> None:
    stub = types.ModuleType("config.settings")
    stub.settings = SimpleNamespace(characters_dir="data/characters")
    sys.modules["config.settings"] = stub


def _install_sandbox_stubs() -> None:
    tick_stub = types.ModuleType("src.sandbox.tick_engine")
    tick_stub.TickEngine = lambda: SimpleNamespace(tick=lambda engine: {})
    sys.modules["src.sandbox.tick_engine"] = tick_stub

    rumor_stub = types.ModuleType("src.sandbox.rumor_spreader")
    rumor_stub.RumorSpreader = lambda: SimpleNamespace(
        spread_tick=lambda all_npc_ids, get_memory_manager: []
    )
    sys.modules["src.sandbox.rumor_spreader"] = rumor_stub


def _build_trace(turn_id: str = "turn-1", session_id: str = "session-1") -> TurnTrace:
    context = SimpleNamespace(
        turn_id=turn_id,
        session_id=session_id,
        character_id="tsundere_sister",
        player_input="你好",
    )
    trace = TurnTrace(
        turn_id=context.turn_id,
        session_id=context.session_id,
        character_id=context.character_id,
        started_at=0.0,
    )
    trace.add_event(
        TurnEvent(
            event_type=EventType.TURN_STARTED,
            turn_id=context.turn_id,
            step_index=0,
            payload={"character_id": context.character_id},
        )
    )
    return trace


def test_process_chat_defaults_to_legacy_pipeline(monkeypatch):
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    engine_module = importlib.import_module("src.engine")

    legacy_response = AgentResponse(
        dialogue="哼，什么事？",
        inner_monologue="这个笨蛋又来了...",
        character_id="tsundere_sister",
        character_name="凌霜",
    )

    engine = engine_module.NPCEngine()
    engine.load_character = lambda character_id, yaml_path=None: engine._characters.setdefault(
        character_id,
        SimpleNamespace(
            id=character_id,
            name="凌霜",
            location="落霞峰",
            initial_relationships=[],
        ),
    ) or engine._characters[character_id]
    engine._pipelines["tsundere_sister"] = SimpleNamespace(
        run=lambda **kwargs: legacy_response
    )
    engine._characters["tsundere_sister"] = SimpleNamespace(
        id="tsundere_sister",
        name="凌霜",
        location="落霞峰",
        initial_relationships=[],
    )

    result = engine.process_chat(
        player_input="你好",
        character_id="tsundere_sister",
        session_id="test-session",
    )

    assert result == legacy_response
    assert engine._state_store.loaded_session_ids == ["test-session"]
    assert engine._state_store.saved_states[0].player.location == "落霞峰"


def test_process_chat_uses_agent_runtime_when_flag_enabled(monkeypatch):
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    engine_module = importlib.import_module("src.engine")

    runtime_result = RuntimeResult(
        dialogue="我在。",
        trace=_build_trace(),
        stop_reason="response_generated",
    )

    class FakeRuntime:
        def __init__(self, policy, tool_executor=None):
            self.policy = policy
            self.tool_executor = tool_executor
            self.calls = []

        def run(self, context):
            self.calls.append(context)
            return runtime_result

    monkeypatch.setattr(engine_module, "AgentRuntime", FakeRuntime)

    engine = engine_module.NPCEngine(use_agent_runtime=True)
    engine.load_character = lambda character_id, yaml_path=None: engine._characters.setdefault(
        character_id,
        SimpleNamespace(
            id=character_id,
            name="凌霜",
            location="落霞峰",
            initial_relationships=[],
        ),
    ) or engine._characters[character_id]
    engine._characters["tsundere_sister"] = SimpleNamespace(
        id="tsundere_sister",
        name="凌霜",
        location="落霞峰",
        initial_relationships=[],
    )

    result = engine.process_chat(
        player_input="你好",
        character_id="tsundere_sister",
        session_id="runtime-session",
    )

    assert result.dialogue == "我在。"
    assert result.character_id == "tsundere_sister"
    assert result.character_name == "凌霜"
    assert result.metadata is None
