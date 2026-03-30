from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

_MISSING = object()
_LOGURU_ORIGINAL = sys.modules.get("loguru", _MISSING)

if _LOGURU_ORIGINAL is _MISSING:
    loguru_stub = types.ModuleType("loguru")
    loguru_stub.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["loguru"] = loguru_stub

from src.models.message import AgentResponse
from src.adapters.memory.composite import CompositeMemoryAdapter
from src.cognition.perception import Perception
from src.models.character import CharacterProfile, Personality
from src.models.game_state import GameState
from src.models.memory import MemoryQueryResult
from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.chat_policy import ChatRuntimePolicy
from src.runtime.runtime_result import RuntimeResult

_MODULE_NAMES = (
    "loguru",
    "src.cognition.pipeline",
    "src.storage.game_state_store",
    "config.settings",
    "src.sandbox.tick_engine",
    "src.sandbox.rumor_spreader",
    "src.engine",
)


@pytest.fixture(autouse=True)
def _restore_module_state():
    snapshot = {name: sys.modules.get(name, _MISSING) for name in _MODULE_NAMES}
    try:
        yield
    finally:
        for name, module in snapshot.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

        if _LOGURU_ORIGINAL is _MISSING:
            sys.modules.pop("loguru", None)
        else:
            sys.modules["loguru"] = _LOGURU_ORIGINAL


class _FakeGameStateStore:
    def __init__(self):
        self.saved_states = []
        self.loaded_session_ids = []

    def load_or_create(self, session_id: str):
        self.loaded_session_ids.append(session_id)
        state = GameState(session_id=session_id)
        state.player.location = "起始地点"
        return state

    def save(self, state):
        self.saved_states.append(state)


class _FakeMemoryManager:
    def __init__(self, character_id: str | None = None):
        self.character_id = character_id
        self.calls = []
        self.relationships = []

    def retrieve(self, query: str, player_id: str = "player_default"):
        self.calls.append((query, player_id))
        return MemoryQueryResult(
            working_memories=[],
            episodic_memories=[],
            semantic_facts=["player 与 npc 的关系为「熟人」"],
            graph_relations=[],
        )

    def init_relationships(self, relationships):
        self.relationships.append(list(relationships))


def _make_character(character_id: str = "tsundere_sister") -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        name="凌霜",
        role="师姐",
        personality=Personality(
            traits=["冷淡", "可靠"],
            speaking_style="简短",
        ),
        location="落霞峰",
        system_prompt="你是凌霜。",
    )


def _import_engine_module():
    sys.modules.pop("src.engine", None)
    return importlib.import_module("src.engine")


def _install_broken_legacy_pipeline_stub() -> None:
    stub = types.ModuleType("src.cognition.pipeline")

    def _raise_on_access(name: str):
        if name == "CognitivePipeline":
            raise AssertionError("legacy pipeline import should stay lazy on runtime path")
        raise AttributeError(name)

    stub.__getattr__ = _raise_on_access
    sys.modules["src.cognition.pipeline"] = stub


def _install_tracking_legacy_pipeline_stub(created_pipelines, run_calls) -> None:
    stub = types.ModuleType("src.cognition.pipeline")

    class FakePipeline:
        def __init__(self, memory_manager):
            created_pipelines.append(memory_manager)
            self._memory_manager = memory_manager

        def run(self, **kwargs):
            run_calls.append(kwargs)
            return AgentResponse(
                dialogue="哼，什么事？",
                inner_monologue="这个笨蛋又来了...",
                character_id=kwargs["character"].id,
                character_name=kwargs["character"].name,
            )

    stub.CognitivePipeline = FakePipeline
    sys.modules["src.cognition.pipeline"] = stub


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
    engine_module = _import_engine_module()

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


def test_global_engine_defaults_to_runtime():
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    engine_module = _import_engine_module()

    engine_module._engine = None

    engine = engine_module.get_engine()

    assert engine._use_agent_runtime is True


def test_engine_import_and_runtime_path_do_not_require_legacy_pipeline(monkeypatch):
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    _install_broken_legacy_pipeline_stub()
    engine_module = _import_engine_module()
    tool_catalog_module = importlib.import_module("src.adapters.tools.catalog")
    tool_executor_module = importlib.import_module("src.adapters.tools.executor")
    inner_monologue_module = importlib.import_module("src.cognition.inner_monologue")
    action_planner_module = importlib.import_module("src.cognition.action_planner")

    class FakeToolExecutor:
        def __init__(self, catalog):
            self.catalog = catalog

    class FakeInnerMonologue:
        def generate(self, context):
            return "先稳住她。"

    class FakeActionPlanner:
        def plan(self, context, inner_monologue: str):
            return SimpleNamespace(
                dialogue="我在。",
                tool_name=None,
                arguments={},
            )

    monkeypatch.setattr(
        engine_module.CharacterProfile,
        "from_yaml",
        classmethod(lambda cls, path: _make_character()),
    )
    monkeypatch.setattr(engine_module, "MemoryManager", _FakeMemoryManager)
    monkeypatch.setattr(tool_catalog_module.ToolCatalog, "load", lambda: object())
    monkeypatch.setattr(tool_executor_module, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(inner_monologue_module, "InnerMonologue", FakeInnerMonologue)
    monkeypatch.setattr(action_planner_module, "ActionPlanner", FakeActionPlanner)

    engine = engine_module.NPCEngine(use_agent_runtime=True)

    result = engine.process_chat(
        player_input="你好",
        character_id="tsundere_sister",
        session_id="runtime-session",
    )

    assert result.dialogue == "我在。"
    assert engine.loaded_characters == ["tsundere_sister"]
    assert "tsundere_sister" not in engine._pipelines
    assert engine._state_store.saved_states[0].player.location == "落霞峰"


def test_legacy_pipeline_is_created_lazily_and_cached(monkeypatch):
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    created_pipelines = []
    run_calls = []
    _install_tracking_legacy_pipeline_stub(created_pipelines, run_calls)
    engine_module = _import_engine_module()

    monkeypatch.setattr(
        engine_module.CharacterProfile,
        "from_yaml",
        classmethod(lambda cls, path: _make_character()),
    )
    monkeypatch.setattr(engine_module, "MemoryManager", _FakeMemoryManager)

    engine = engine_module.NPCEngine()
    engine.load_character("tsundere_sister")

    assert engine.loaded_characters == ["tsundere_sister"]
    assert created_pipelines == []
    assert "tsundere_sister" not in engine._pipelines

    first_result = engine.process_chat(
        player_input="你好",
        character_id="tsundere_sister",
        session_id="legacy-session",
    )
    second_result = engine.process_chat(
        player_input="又见面了",
        character_id="tsundere_sister",
        session_id="legacy-session",
    )

    assert first_result.dialogue == "哼，什么事？"
    assert second_result.dialogue == "哼，什么事？"
    assert len(created_pipelines) == 1
    assert len(run_calls) == 2
    assert run_calls[0]["player_input"] == "你好"
    assert run_calls[1]["player_input"] == "又见面了"


def test_process_chat_uses_agent_runtime_when_flag_enabled(monkeypatch):
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    engine_module = _import_engine_module()
    tool_catalog_module = importlib.import_module("src.adapters.tools.catalog")
    tool_executor_module = importlib.import_module("src.adapters.tools.executor")
    inner_monologue_module = importlib.import_module("src.cognition.inner_monologue")
    action_planner_module = importlib.import_module("src.cognition.action_planner")

    seen_contexts = []

    class FakeToolExecutor:
        def __init__(self, catalog):
            self.catalog = catalog

    class FakeInnerMonologue:
        def generate(self, context):
            seen_contexts.append(context)
            return "先稳住她。"

    class FakeActionPlanner:
        def plan(self, context, inner_monologue: str):
            seen_contexts.append((context, inner_monologue))
            return SimpleNamespace(
                dialogue="我在。",
                tool_name=None,
                arguments={},
            )

    monkeypatch.setattr(tool_catalog_module.ToolCatalog, "load", lambda: object())
    monkeypatch.setattr(tool_executor_module, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(inner_monologue_module, "InnerMonologue", FakeInnerMonologue)
    monkeypatch.setattr(action_planner_module, "ActionPlanner", FakeActionPlanner)

    engine = engine_module.NPCEngine(use_agent_runtime=True)
    character = _make_character()
    engine._characters[character.id] = character
    engine._memory_managers[character.id] = _FakeMemoryManager()

    result = engine.process_chat(
        player_input="你好",
        character_id=character.id,
        session_id="runtime-session",
    )

    assert result.dialogue == "我在。"
    assert result.emotion == "neutral"
    assert result.inner_monologue is None
    assert result.tool_calls == []
    assert result.state_changes == {}
    assert result.character_id == "tsundere_sister"
    assert result.character_name == "凌霜"
    assert result.metadata is None
    assert seen_contexts[0].game_state.session_id == "runtime-session"
    assert seen_contexts[0].game_state.player.location == "落霞峰"
    assert "落霞峰" in seen_contexts[0].environment_desc
    assert seen_contexts[1] == (seen_contexts[0], "先稳住她。")


def test_get_or_create_runtime_builds_chat_runtime_with_real_collaborators(monkeypatch):
    _install_loguru_stub()
    _install_game_state_store_stub()
    _install_settings_stub()
    _install_sandbox_stubs()
    engine_module = _import_engine_module()
    tool_catalog_module = importlib.import_module("src.adapters.tools.catalog")
    tool_executor_module = importlib.import_module("src.adapters.tools.executor")
    inner_monologue_module = importlib.import_module("src.cognition.inner_monologue")
    action_planner_module = importlib.import_module("src.cognition.action_planner")

    fake_catalog = object()

    class FakeToolExecutor:
        def __init__(self, catalog):
            self.catalog = catalog

    class FakeInnerMonologue:
        pass

    class FakeActionPlanner:
        pass

    monkeypatch.setattr(tool_catalog_module.ToolCatalog, "load", lambda: fake_catalog)
    monkeypatch.setattr(tool_executor_module, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(inner_monologue_module, "InnerMonologue", FakeInnerMonologue)
    monkeypatch.setattr(action_planner_module, "ActionPlanner", FakeActionPlanner)

    engine = engine_module.NPCEngine(use_agent_runtime=True)
    character = _make_character()
    memory_manager = _FakeMemoryManager()
    engine._characters[character.id] = character
    engine._memory_managers[character.id] = memory_manager

    runtime = engine._get_or_create_runtime(character.id)

    assert isinstance(runtime.policy, ChatRuntimePolicy)
    assert isinstance(runtime.tool_executor, FakeToolExecutor)
    assert runtime.tool_executor.catalog is fake_catalog
    assert isinstance(runtime.memory_adapter, CompositeMemoryAdapter)
    assert isinstance(runtime.perception, Perception)
    assert isinstance(runtime.reflector, FakeInnerMonologue)
    assert isinstance(runtime.action_planner, FakeActionPlanner)
    assert runtime.character == character
