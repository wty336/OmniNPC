from pathlib import Path
import importlib
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.game_state import GameState
from src.models.message import AgentResponse
from src.observability.trace import TurnTrace
from src.runtime.runtime_result import RuntimeResult


class _FakeStateStore:
    def __init__(self):
        self.saved_states = []

    def load_or_create(self, session_id: str) -> GameState:
        return GameState(session_id=session_id)

    def save(self, state: GameState) -> None:
        self.saved_states.append(state)


class _FakeLegacyPipeline:
    def run(self, **kwargs) -> AgentResponse:
        character = kwargs["character"]
        return AgentResponse(
            dialogue="哼，什么事？",
            inner_monologue="这个笨蛋又来了...",
            character_id=character.id,
            character_name=character.name,
        )


class _FakeRuntime:
    def run(self, turn_context, game_state: GameState) -> RuntimeResult:
        trace = TurnTrace.start(turn_context)
        trace.finish()
        return RuntimeResult(
            dialogue="我在。",
            trace=trace,
            stop_reason="response_generated",
        )


def _install_loguru_stub() -> None:
    if "loguru" in sys.modules:
        return

    stub = types.ModuleType("loguru")
    stub.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["loguru"] = stub


def _install_settings_stub() -> None:
    stub = types.ModuleType("config.settings")
    stub.settings = SimpleNamespace(characters_dir="data/characters")
    sys.modules["config.settings"] = stub


def _install_game_state_store_stub() -> None:
    stub = types.ModuleType("src.storage.game_state_store")
    stub.GameStateStore = _FakeStateStore
    sys.modules["src.storage.game_state_store"] = stub


def _install_sandbox_stubs() -> None:
    tick_stub = types.ModuleType("src.sandbox.tick_engine")
    tick_stub.TickEngine = lambda: SimpleNamespace(tick=lambda engine: {})
    sys.modules["src.sandbox.tick_engine"] = tick_stub

    rumor_stub = types.ModuleType("src.sandbox.rumor_spreader")
    rumor_stub.RumorSpreader = lambda: SimpleNamespace(
        spread_tick=lambda all_npc_ids, get_memory_manager: []
    )
    sys.modules["src.sandbox.rumor_spreader"] = rumor_stub


def _import_engine_module():
    _install_loguru_stub()
    _install_settings_stub()
    _install_game_state_store_stub()
    _install_sandbox_stubs()
    sys.modules.pop("src.engine", None)
    return importlib.import_module("src.engine")


def _build_engine(use_agent_runtime: bool):
    engine_module = _import_engine_module()
    character_id = "tsundere_sister"
    character = SimpleNamespace(
        id=character_id,
        name="凌霜",
        location="落霞峰",
    )

    engine = engine_module.NPCEngine(use_agent_runtime=use_agent_runtime)
    engine._characters[character_id] = character
    engine._state_store = _FakeStateStore()

    if use_agent_runtime:
        engine._runtimes[character_id] = _FakeRuntime()
    else:
        engine._pipelines[character_id] = _FakeLegacyPipeline()

    return engine


def test_runtime_and_legacy_paths_return_same_response_shape():
    legacy = _build_engine(use_agent_runtime=False)
    runtime = _build_engine(use_agent_runtime=True)

    legacy_response = legacy.process_chat("师姐救我", "tsundere_sister", "s1")
    runtime_response = runtime.process_chat("师姐救我", "tsundere_sister", "s1")

    assert set(legacy_response.model_dump().keys()) == set(
        runtime_response.model_dump().keys()
    )
    assert runtime_response.dialogue
    assert runtime_response.character_id == legacy_response.character_id
    assert runtime_response.character_name == legacy_response.character_name
    assert isinstance(runtime_response.dialogue, str)
    assert runtime_response.tool_calls is not None
    assert runtime_response.state_changes is not None
