# OmniNPC Agent Harness Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pipeline-centered execution path in OmniNPC with a limited-loop AgentRuntime that makes agent behavior explicit, observable, and testable while preserving the current chat use case.

**Architecture:** Introduce `runtime`, `observability`, and `adapters` as first-class layers. `NPCEngine` becomes a composition root, cognition modules become callable abilities instead of the controlling workflow, and the old `CognitivePipeline` is quarantined as a temporary legacy path instead of remaining the system center.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Loguru, pytest, ChromaDB, NetworkX

---

### Task 1: Create Runtime And Trace Contracts

**Files:**
- Create: `src/runtime/__init__.py`
- Create: `src/runtime/turn_context.py`
- Create: `src/runtime/runtime_result.py`
- Create: `src/runtime/step_budget.py`
- Create: `src/observability/__init__.py`
- Create: `src/observability/events.py`
- Create: `src/observability/trace.py`
- Test: `tests/runtime/test_runtime_contracts.py`

**Status:** Completed, not committed

**Actual verification result:**
- `pytest tests/runtime/test_runtime_contracts.py -v`
- Result: `9 passed`
- Environment note: the test keeps a local `sys.path` bootstrap so the original `pytest` console-script command resolves `src` correctly in this shell. Project-level pytest/import configuration was intentionally left unchanged.

**Actual implementation notes discovered during Task 1 review:**
- Contract models now reject unexpected fields via `extra="forbid"`.
- `StepBudget` now rejects invalid construction where `used_steps > max_steps` and raises when consuming past the budget.
- `TurnTrace.add_event()` now rejects events whose `turn_id` does not match the trace.

- [x] **Step 1: Write the failing test**

```python
from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.runtime_result import RuntimeResult
from src.runtime.step_budget import StepBudget
from src.runtime.turn_context import TurnContext


def test_turn_context_and_trace_defaults():
    context = TurnContext(
        turn_id="turn-1",
        session_id="session-1",
        character_id="tsundere_sister",
        player_input="师姐，帮帮我。",
    )

    trace = TurnTrace.start(context)
    trace.add_event(
        TurnEvent(
            event_type=EventType.TURN_STARTED,
            turn_id=context.turn_id,
            step_index=0,
            payload={"character_id": context.character_id},
        )
    )

    result = RuntimeResult(
        dialogue="先躲起来。",
        trace=trace,
        stop_reason="response_generated",
    )

    assert context.max_steps == 4
    assert StepBudget(max_steps=4).remaining_steps == 4
    assert trace.turn_id == "turn-1"
    assert trace.events[0].event_type is EventType.TURN_STARTED
    assert result.stop_reason == "response_generated"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_runtime_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.runtime` or `src.observability`

- [x] **Step 3: Write minimal implementation**

```python
# src/runtime/turn_context.py
from pydantic import BaseModel, Field


class TurnContext(BaseModel):
    turn_id: str
    session_id: str
    character_id: str
    player_input: str
    max_steps: int = Field(default=4, ge=1)
    debug: bool = False
```

```python
# src/runtime/step_budget.py
from pydantic import BaseModel, Field


class StepBudget(BaseModel):
    max_steps: int = Field(default=4, ge=1)
    used_steps: int = Field(default=0, ge=0)

    @property
    def remaining_steps(self) -> int:
        return max(self.max_steps - self.used_steps, 0)

    def consume(self) -> None:
        self.used_steps += 1
```

```python
# src/observability/events.py
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    TURN_STARTED = "turn_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    ACTION_SELECTED = "action_selected"
    TURN_FINISHED = "turn_finished"


class TurnEvent(BaseModel):
    event_type: EventType
    turn_id: str
    step_index: int = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
```

```python
# src/observability/trace.py
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from src.observability.events import TurnEvent
from src.runtime.turn_context import TurnContext


class TurnTrace(BaseModel):
    turn_id: str
    session_id: str
    character_id: str
    started_at: float
    ended_at: float | None = None
    events: list[TurnEvent] = Field(default_factory=list)

    @classmethod
    def start(cls, context: TurnContext) -> "TurnTrace":
        return cls(
            turn_id=context.turn_id,
            session_id=context.session_id,
            character_id=context.character_id,
            started_at=time.time(),
        )

    def add_event(self, event: TurnEvent) -> None:
        self.events.append(event)

    def finish(self) -> None:
        self.ended_at = time.time()
```

```python
# src/runtime/runtime_result.py
from pydantic import BaseModel

from src.observability.trace import TurnTrace


class RuntimeResult(BaseModel):
    dialogue: str
    trace: TurnTrace
    stop_reason: str
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/runtime/test_runtime_contracts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

Deferred by user request: Task 1 remains uncommitted for now.

```bash
git add tests/runtime/test_runtime_contracts.py src/runtime src/observability
git commit -m "feat: add runtime and trace contracts"
```

### Task 2: Implement Limited-Loop AgentRuntime

**Files:**
- Create: `src/runtime/actions.py`
- Create: `src/runtime/stop_conditions.py`
- Create: `src/runtime/agent_runtime.py`
- Create: `src/runtime/policies.py`
- Modify: `src/runtime/__init__.py`
- Test: `tests/runtime/test_agent_runtime.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_agent_runtime.py -v`
- Result: `3 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py -v`
- Result: `12 passed`

**Actual implementation notes discovered during Task 2 review:**
- `src/runtime/__init__.py` was reduced to a minimal package file to avoid circular imports from eager re-exports.
- `DefaultRuntimePolicy` now emits a non-empty `respond` action only.
- `RuntimeAction` now enforces that `respond` actions carry a non-empty `payload["dialogue"]`.
- The scoped tests now cover:
  - default response propagation
  - budget exhaustion without response
  - invalid empty `respond` actions

- [x] **Step 1: Write the failing test**

```python
from src.runtime.actions import RuntimeAction
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.turn_context import TurnContext


class StubPolicy:
    def __init__(self):
        self._actions = [
            RuntimeAction(action_type="retrieve_memory"),
            RuntimeAction(action_type="reflect"),
            RuntimeAction(action_type="respond", payload={"dialogue": "先躲起来。"}),
        ]

    def next_action(self, state):
        return self._actions.pop(0)


def test_agent_runtime_stops_after_response():
    runtime = AgentRuntime(policy=StubPolicy())
    result = runtime.run(
        TurnContext(
            turn_id="turn-2",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="师姐救我",
        )
    )

    assert result.dialogue == "先躲起来。"
    assert result.stop_reason == "response_generated"
    assert len(result.trace.events) >= 4
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_agent_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` for `AgentRuntime` / `RuntimeAction`

- [x] **Step 3: Write minimal implementation**

```python
# src/runtime/actions.py
from typing import Any

from pydantic import BaseModel, Field


class RuntimeAction(BaseModel):
    action_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
```

```python
# src/runtime/stop_conditions.py
from src.runtime.step_budget import StepBudget


def should_stop(dialogue: str, budget: StepBudget) -> str | None:
    if dialogue:
        return "response_generated"
    if budget.remaining_steps == 0:
        return "step_budget_exhausted"
    return None
```

```python
# src/runtime/agent_runtime.py
from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.runtime_result import RuntimeResult
from src.runtime.step_budget import StepBudget
from src.runtime.stop_conditions import should_stop


class AgentRuntime:
    def __init__(self, policy):
        self.policy = policy

    def run(self, context):
        trace = TurnTrace.start(context)
        budget = StepBudget(max_steps=context.max_steps)
        dialogue = ""
        state = {"context": context, "artifacts": {}}

        trace.add_event(TurnEvent(event_type=EventType.TURN_STARTED, turn_id=context.turn_id, step_index=0))

        while budget.remaining_steps > 0:
            step_index = budget.used_steps + 1
            trace.add_event(TurnEvent(event_type=EventType.STEP_STARTED, turn_id=context.turn_id, step_index=step_index))
            action = self.policy.next_action(state)
            trace.add_event(
                TurnEvent(
                    event_type=EventType.ACTION_SELECTED,
                    turn_id=context.turn_id,
                    step_index=step_index,
                    payload={"action_type": action.action_type},
                )
            )

            if action.action_type == "respond":
                dialogue = action.payload["dialogue"]

            budget.consume()
            trace.add_event(TurnEvent(event_type=EventType.STEP_COMPLETED, turn_id=context.turn_id, step_index=step_index))

            stop_reason = should_stop(dialogue=dialogue, budget=budget)
            if stop_reason:
                trace.add_event(
                    TurnEvent(
                        event_type=EventType.TURN_FINISHED,
                        turn_id=context.turn_id,
                        step_index=step_index,
                        payload={"stop_reason": stop_reason},
                    )
                )
                trace.finish()
                return RuntimeResult(dialogue=dialogue, trace=trace, stop_reason=stop_reason)

        trace.finish()
        return RuntimeResult(dialogue=dialogue, trace=trace, stop_reason="step_budget_exhausted")
```

```python
# src/runtime/policies.py
from src.runtime.actions import RuntimeAction


class DefaultRuntimePolicy:
    def next_action(self, state):
        if "memory" not in state["artifacts"]:
            return RuntimeAction(action_type="retrieve_memory")
        if "reflection" not in state["artifacts"]:
            return RuntimeAction(action_type="reflect")
        return RuntimeAction(action_type="respond", payload={"dialogue": "我在。先别乱动。"})
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/runtime/test_agent_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

Deferred by user request: Task 2 remains uncommitted for now.

```bash
git add tests/runtime/test_agent_runtime.py src/runtime
git commit -m "feat: add limited-loop agent runtime"
```

### Task 3: Extract Model Adapter And Model Call Events

**Files:**
- Create: `src/adapters/__init__.py`
- Create: `src/adapters/llm/__init__.py`
- Create: `src/adapters/llm/base.py`
- Create: `src/adapters/llm/ark_adapter.py`
- Modify: `src/llm/client.py`
- Modify: `src/observability/events.py`
- Test: `tests/adapters/test_model_adapter.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/adapters/test_model_adapter.py -v`
- Result: `5 passed`
- Regression check: `conda activate plante; python -m pytest tests/adapters/test_model_adapter.py tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py -v`
- Result: `17 passed`

**Actual implementation notes discovered during Task 3 review:**
- `ModelRequest` / `ModelResponse` now follow the strict contract style (`extra="forbid"`).
- `ArkModelAdapter` remains a thin wrapper over `get_llm_client().chat(...)`.
- `src/llm/client.py` now includes `model_name` in the returned chat payload.
- `src/llm/client.py` now preserves explicit zero-value overrides such as `temperature=0.0` and `max_tokens=0`.
- `EventType` now includes `MODEL_CALLED`, but no event emission was added yet.

- [x] **Step 1: Write the failing test**

```python
from src.adapters.llm.base import ModelRequest, ModelResponse


class FakeAdapter:
    def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="收到。",
            usage={"total_tokens": 10},
            model_name="fake-model",
        )


def test_model_adapter_contract():
    adapter = FakeAdapter()
    request = ModelRequest(
        purpose="reflect",
        messages=[{"role": "user", "content": "你好"}],
    )
    response = adapter.complete(request)

    assert response.content == "收到。"
    assert response.usage["total_tokens"] == 10
    assert response.model_name == "fake-model"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/adapters/test_model_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.adapters.llm.base`

- [x] **Step 3: Write minimal implementation**

```python
# src/adapters/llm/base.py
from typing import Any

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    purpose: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None


class ModelResponse(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    model_name: str = ""
```

```python
# src/adapters/llm/ark_adapter.py
from src.adapters.llm.base import ModelRequest, ModelResponse
from src.llm.client import get_llm_client


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
            model_name="ark",
        )
```

```python
# src/observability/events.py
class EventType(str, Enum):
    TURN_STARTED = "turn_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    ACTION_SELECTED = "action_selected"
    MODEL_CALLED = "model_called"
    TURN_FINISHED = "turn_finished"
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/adapters/test_model_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

Deferred by user request: Task 3 remains uncommitted for now.

```bash
git add tests/adapters/test_model_adapter.py src/adapters src/observability/events.py src/llm/client.py
git commit -m "feat: add model adapter contract"
```

### Task 4: Move Tool Discovery And Execution Under Runtime Control

**Files:**
- Create: `src/adapters/tools/__init__.py`
- Create: `src/adapters/tools/types.py`
- Create: `src/adapters/tools/catalog.py`
- Create: `src/adapters/tools/executor.py`
- Modify: `src/tools/base.py`
- Modify: `src/tools/state_updater.py`
- Modify: `src/tools/item_manager.py`
- Test: `tests/adapters/test_tool_executor.py`
- Test: `tests/runtime/test_runtime_tool_flow.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py -v`
- Result: `6 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py tests/adapters/test_model_adapter.py tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py -v`
- Result: `23 passed`

**Actual implementation notes discovered during Task 4 review:**
- `ToolCatalog` now supports both direct `register/get` and `load()` from the legacy tool registry.
- `AgentRuntime` was necessarily extended to accept an optional `tool_executor`, execute `propose_tool` actions, and continue to `respond` instead of terminating immediately after tool execution.
- To keep the default runtime tool path usable without touching `engine`, `AgentRuntime` now seeds a minimal in-memory `GameState(session_id=context.session_id)` and writes `last_tool_execution` back into runtime state for the next policy step.
- The planned edits to `src/tools/base.py`, `src/tools/state_updater.py`, and `src/tools/item_manager.py` were not needed; the adapter layer was able to wrap the legacy registry with lower intrusion.

- [x] **Step 1: Write the failing tests**

```python
from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.executor import ToolExecutor


def echo_tool(game_state, value: str):
    return {"echo": value}


def test_tool_catalog_and_executor():
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

    executor = ToolExecutor(catalog=catalog)
    result = executor.execute("echo_tool", {"value": "x"}, game_state={})

    assert result.success is True
    assert result.output == {"echo": "x"}
```

```python
from src.runtime.actions import RuntimeAction
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.turn_context import TurnContext
from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.executor import ToolExecutor


class ToolPolicy:
    def __init__(self):
        self._actions = [
            RuntimeAction(action_type="propose_tool", payload={"tool_name": "echo_tool", "arguments": {"value": "x"}}),
            RuntimeAction(action_type="respond", payload={"dialogue": "办好了。"}),
        ]

    def next_action(self, state):
        return self._actions.pop(0)


def test_runtime_executes_tool_before_response():
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
    runtime = AgentRuntime(policy=ToolPolicy(), tool_executor=ToolExecutor(catalog))
    result = runtime.run(
        TurnContext(
            turn_id="turn-3",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="做点什么",
        )
    )

    assert any(event.payload.get("tool_name") == "echo_tool" for event in result.trace.events)
    assert result.dialogue == "办好了。"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py -v`
Expected: FAIL with missing tool adapter classes or incomplete runtime tool handling

- [x] **Step 3: Write minimal implementation**

```python
# src/adapters/tools/types.py
from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionResult(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    success: bool = True
    error: str | None = None
```

```python
# src/adapters/tools/catalog.py
class ToolCatalog:
    def __init__(self):
        self._registry = {}
        self._definitions = []

    def register(self, name, description, parameters, func):
        self._registry[name] = func
        self._definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    def definitions(self):
        return list(self._definitions)

    def get(self, name):
        return self._registry.get(name)
```

```python
# src/adapters/tools/executor.py
from src.adapters.tools.types import ToolExecutionResult


class ToolExecutor:
    def __init__(self, catalog):
        self.catalog = catalog

    def execute(self, tool_name, arguments, game_state):
        func = self.catalog.get(tool_name)
        if func is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            output = func(game_state=game_state, **arguments)
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                output=output,
                success=True,
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=str(exc),
            )
```

```python
# src/tools/base.py
GLOBAL_TOOL_CATALOG = ToolCatalog()


def register_tool(name: str, description: str, parameters: dict):
    def decorator(func):
        GLOBAL_TOOL_CATALOG.register(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        return func
    return decorator


def get_tool_catalog() -> ToolCatalog:
    return GLOBAL_TOOL_CATALOG
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py src/adapters/tools src/tools
git commit -m "feat: move tool execution under runtime control"
```

### Task 5: Convert Memory And Cognition Modules Into Runtime Abilities

**Files:**
- Create: `src/adapters/memory/__init__.py`
- Create: `src/adapters/memory/base.py`
- Create: `src/adapters/memory/composite.py`
- Modify: `src/memory/memory_manager.py`
- Modify: `src/cognition/perception.py`
- Modify: `src/cognition/inner_monologue.py`
- Modify: `src/cognition/action_generator.py`
- Test: `tests/adapters/test_memory_adapter.py`
- Test: `tests/cognition/test_perception_ability.py`
- Test: `tests/cognition/test_reflection_and_action_policy.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py -v`
- Result: `5 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py tests/adapters/test_model_adapter.py tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py tests/test_cognition.py -v`
- Result: `34 passed`

**Actual implementation notes discovered during Task 5 review:**
- `Perception` now accepts either an injected `memory_adapter` or the legacy `memory_manager` argument; the old `Perception(memory_manager)` construction path remains valid.
- `InnerMonologue` and `ActionGenerator` now depend on `model_adapter.complete(ModelRequest(...))`, but keep their no-arg constructors by defaulting to `ArkModelAdapter()`.
- `ActionGenerator` preserved its existing tool-call execution path; Task 5 intentionally stopped at model/ability injection and did not move tool execution out of cognition.
- `src/memory/memory_manager.py` did not require changes; `CompositeMemoryAdapter` wraps it with lower intrusion.

- [x] **Step 1: Write the failing tests**

```python
from src.adapters.memory.base import MemoryAdapter, MemorySnapshot


class FakeMemoryAdapter(MemoryAdapter):
    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        return MemorySnapshot(
            working_memories=[],
            episodic_memories=[],
            semantic_facts=["player 与 npc 的关系为「友人」(权重: 0.6)"],
            graph_relations=[],
        )

    def consolidate(self, **kwargs) -> None:
        self.last_consolidated = kwargs


def test_memory_adapter_contract():
    adapter = FakeMemoryAdapter()
    snapshot = adapter.retrieve("你好")
    assert "友人" in snapshot.semantic_facts[0]
```

```python
from src.cognition.perception import Perception
from src.models.character import CharacterProfile, Personality
from src.models.game_state import GameState


def test_perception_reads_from_adapter():
    character = CharacterProfile(
        id="npc",
        name="凌霜",
        role="师姐",
        personality=Personality(traits=["冷淡"], speaking_style="简短"),
        system_prompt="你是凌霜。",
    )
    game_state = GameState(session_id="session-1")
    perception = Perception(memory_adapter=FakeMemoryAdapter())

    context = perception.perceive("你好", character, game_state)

    assert "友人" in context.memory_result.semantic_facts[0]
```

```python
from src.adapters.llm.base import ModelResponse
from src.cognition.action_generator import ActionGenerator
from src.cognition.inner_monologue import InnerMonologue
from types import SimpleNamespace


class FakeModelAdapter:
    def __init__(self):
        self.calls = []

    def complete(self, request):
        self.calls.append(request.purpose)
        if request.purpose == "reflect":
            return ModelResponse(content="他看起来很慌，我先稳住他。")
        return ModelResponse(content="别慌，先站到我身后。")


def test_reflection_and_action_use_model_adapter():
    model = FakeModelAdapter()
    context = SimpleNamespace(
        player_input="救我",
        environment_desc="当前地点：大殿",
        character=SimpleNamespace(
            id="npc",
            name="凌霜",
            role="师姐",
            system_prompt="你是凌霜。",
        ),
        game_state=SimpleNamespace(player=SimpleNamespace(player_id="player_default")),
        memory_result=SimpleNamespace(working_memories=[], episodic_memories=[], semantic_facts=[]),
    )
    monologue = InnerMonologue(model_adapter=model).generate(context=context)
    response = ActionGenerator(model_adapter=model).generate(context=context, inner_monologue=monologue)

    assert monologue.startswith("他看起来")
    assert response.dialogue.startswith("别慌")
    assert model.calls == ["reflect", "respond"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py -v`
Expected: FAIL with missing adapter classes or direct `get_llm_client()` dependencies

- [x] **Step 3: Write minimal implementation**

```python
# src/adapters/memory/base.py
from abc import ABC, abstractmethod

from src.models.memory import MemoryQueryResult


class MemorySnapshot(MemoryQueryResult):
    pass


class MemoryAdapter(ABC):
    @abstractmethod
    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        raise NotImplementedError

    @abstractmethod
    def consolidate(self, **kwargs) -> None:
        raise NotImplementedError
```

```python
# src/adapters/memory/composite.py
from src.adapters.memory.base import MemoryAdapter, MemorySnapshot
from src.memory.memory_manager import MemoryManager


class CompositeMemoryAdapter(MemoryAdapter):
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    def retrieve(self, query: str, player_id: str = "player_default") -> MemorySnapshot:
        result = self.memory_manager.retrieve(query=query, player_id=player_id)
        return MemorySnapshot(**result.model_dump())

    def consolidate(self, **kwargs) -> None:
        self.memory_manager.consolidate(**kwargs)
```

```python
# src/cognition/perception.py
class Perception:
    def __init__(self, memory_adapter):
        self._memory = memory_adapter
```

```python
# src/cognition/inner_monologue.py
from src.adapters.llm.base import ModelRequest


class InnerMonologue:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def generate(self, context):
        prompt = (
            f"你是{context.character.name}，角色是{context.character.role}。\n"
            f"当前环境：{context.environment_desc}\n"
            f"玩家输入：{context.player_input}\n"
            "请只输出一段内心独白。"
        )
        result = self.model_adapter.complete(
            ModelRequest(
                purpose="reflect",
                messages=[
                    {"role": "system", "content": "你是一个角色扮演内心独白生成器。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=300,
            )
        )
        return result.content.strip()
```

```python
# src/cognition/action_generator.py
from src.adapters.llm.base import ModelRequest


class ActionGenerator:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def generate(self, context, inner_monologue):
        messages = [
            {"role": "system", "content": context.character.system_prompt},
            {"role": "user", "content": context.player_input},
            {"role": "assistant", "content": inner_monologue},
        ]
        result = self.model_adapter.complete(
            ModelRequest(
                purpose="respond",
                messages=messages,
                temperature=0.8,
            )
        )
        return AgentResponse(
            dialogue=result.content,
            inner_monologue=inner_monologue,
            character_id=context.character.id,
            character_name=context.character.name,
        )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py src/adapters/memory src/cognition src/memory/memory_manager.py
git commit -m "refactor: convert cognition modules to runtime abilities"
```

### Task 6: Integrate AgentRuntime Into Engine And Preserve Chat Path

**Files:**
- Modify: `src/engine.py`
- Modify: `src/api/routes/chat.py`
- Modify: `src/models/message.py`
- Create: `src/cognition/legacy_pipeline.py`
- Modify: `src/cognition/pipeline.py`
- Test: `tests/test_api.py`
- Test: `tests/runtime/test_engine_runtime_integration.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py -v`
- Result: `2 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py tests/adapters/test_model_adapter.py tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py tests/test_cognition.py tests/runtime/test_engine_runtime_integration.py -v`
- Result: `36 passed`
- Environment note: `tests/test_api.py` could not be executed in `plante` because the environment currently lacks `fastapi`, so API-path verification for this task remained limited to the minimal synchronous compatibility change in `src/api/routes/chat.py`.

**Actual implementation notes discovered during Task 6 review:**
- `NPCEngine` now accepts `use_agent_runtime: bool = False`; default behavior still goes through the legacy `CognitivePipeline`.
- Dual-path routing lives entirely inside `src/engine.py`: legacy pipeline remains the default path, while the runtime path is lazily created per character via `_runtimes`.
- `RuntimeResult` is adapted back into the existing `AgentResponse` shape without adding new response fields or changing the API schema.
- To preserve the synchronous `NPCEngine.process_chat()` interface, `src/api/routes/chat.py` received a one-line compatibility fix removing `await` from the engine call.
- The planned edits to `src/models/message.py`, `src/cognition/legacy_pipeline.py`, and `src/cognition/pipeline.py` were intentionally skipped because the dual-path engine integration did not require them.

- [x] **Step 1: Write the failing tests**

```python
from unittest.mock import patch

from src.runtime.runtime_result import RuntimeResult
from src.observability.trace import TurnTrace
from src.runtime.turn_context import TurnContext


def test_engine_uses_agent_runtime():
    with patch("src.engine.AgentRuntime") as runtime_cls:
        runtime_instance = runtime_cls.return_value
        runtime_instance.run.return_value = RuntimeResult(
            dialogue="先退后。",
            trace=TurnTrace.start(
                TurnContext(
                    turn_id="test:tsundere_sister",
                    session_id="test",
                    character_id="tsundere_sister",
                    player_input="救我",
                )
            ),
            stop_reason="response_generated",
        )

        from src.engine import NPCEngine

        engine = NPCEngine()
        response = engine.process_chat("救我", "tsundere_sister", "test")

        assert response.dialogue == "先退后。"
        runtime_instance.run.assert_called_once()
```

```python
def test_chat_endpoint_format(client):
    response = client.post("/api/v1/chat", json={
        "player_input": "你好",
        "character_id": "tsundere_sister",
        "session_id": "test",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "dialogue" in data["data"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py tests/runtime/test_engine_runtime_integration.py -v`
Expected: FAIL because `NPCEngine` still instantiates `CognitivePipeline`

- [x] **Step 3: Write minimal implementation**

```python
# src/engine.py
from src.adapters.llm.ark_adapter import ArkModelAdapter
from src.adapters.memory.composite import CompositeMemoryAdapter
from src.adapters.tools.executor import ToolExecutor
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.policies import DefaultRuntimePolicy
from src.runtime.turn_context import TurnContext
from src.tools.base import get_tool_catalog


class NPCEngine:
    def load_character(self, character_id: str, yaml_path: str | None = None):
        memory = MemoryManager(character_id)
        memory_adapter = CompositeMemoryAdapter(memory)
        model_adapter = ArkModelAdapter()
        tool_executor = ToolExecutor(get_tool_catalog())
        self._pipelines[character_id] = AgentRuntime(
            policy=DefaultRuntimePolicy(),
            model_adapter=model_adapter,
            memory_adapter=memory_adapter,
            tool_executor=tool_executor,
        )

    def process_chat(self, player_input: str, character_id: str, session_id: str = "default_session"):
        runtime = self._pipelines[character_id]
        turn = TurnContext(
            turn_id=f"{session_id}:{character_id}",
            session_id=session_id,
            character_id=character_id,
            player_input=player_input,
        )
        runtime_result = runtime.run(turn_context=turn, character=character, game_state=game_state)
        return AgentResponse(
            dialogue=runtime_result.dialogue,
            character_id=character.id,
            character_name=character.name,
            metadata={"stop_reason": runtime_result.stop_reason},
        )
```

```python
# src/models/message.py
class AgentResponse(BaseModel):
    trace: Optional[dict[str, Any]] = Field(
        default=None,
        description="调试模式下返回的 turn trace 摘要",
    )
```

```python
# src/cognition/legacy_pipeline.py
from src.cognition.pipeline import CognitivePipeline

__all__ = ["CognitivePipeline"]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py tests/runtime/test_engine_runtime_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py tests/runtime/test_engine_runtime_integration.py src/engine.py src/api/routes/chat.py src/models/message.py src/cognition/legacy_pipeline.py src/cognition/pipeline.py
git commit -m "refactor: route chat flow through agent runtime"
```

### Task 7: Add Trace-Centric Regression Tests And Legacy Quarantine

**Files:**
- Create: `tests/runtime/test_trace_sequence.py`
- Create: `tests/runtime/test_stop_conditions.py`
- Modify: `tests/test_cognition.py`
- Modify: `tests/test_phase2.py`
- Modify: `src/sandbox/tick_engine.py`
- Modify: `src/sandbox/rumor_spreader.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_trace_sequence.py tests/runtime/test_stop_conditions.py tests/test_phase2.py -v`
- Result: `14 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py tests/runtime/test_trace_sequence.py tests/runtime/test_stop_conditions.py tests/adapters/test_model_adapter.py tests/adapters/test_tool_executor.py tests/runtime/test_runtime_tool_flow.py tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py tests/test_cognition.py tests/test_phase2.py tests/runtime/test_engine_runtime_integration.py -v`
- Result: `50 passed`

**Actual implementation notes discovered during Task 7 review:**
- Runtime side stayed test-only: the new trace and stop-condition coverage passed without changing `src/runtime/*`.
- `TickEngine` now prefers a public `engine.iter_memory_managers()` hook when available, and only falls back to the old private `_memory_managers` path otherwise.
- `RumorSpreader` now supports a lighter `store_memory_item(...)` sink in addition to the legacy `episodic.store(...)` path, reducing the hard assumption that every target is a full `MemoryManager`.
- `NPCEngine` received minimal public memory-manager accessors so sandbox code can stop reaching directly into its private internals.

- [x] **Step 1: Write the failing tests**

```python
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.turn_context import TurnContext
from src.runtime.actions import RuntimeAction


class FakeRuntimeDependency:
    def retrieve(self, *args, **kwargs):
        return {"memory": []}

    def complete(self, *args, **kwargs):
        return {"content": "收到"}


class TracePolicy:
    def __init__(self):
        self.actions = [
            RuntimeAction(action_type="retrieve_memory"),
            RuntimeAction(action_type="respond", payload={"dialogue": "我来处理。"}),
        ]

    def next_action(self, state):
        return self.actions.pop(0)


def test_trace_contains_expected_event_order():
    runtime = AgentRuntime(policy=TracePolicy())
    result = runtime.run(
        TurnContext(
            turn_id="turn-4",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="师姐，快帮我。",
        )
    )

    event_types = [event.event_type.value for event in result.trace.events]
    assert event_types[0] == "turn_started"
    assert "action_selected" in event_types
    assert event_types[-1] == "turn_finished"
```

```python
from src.runtime.actions import RuntimeAction


class NeverRespondPolicy:
    def next_action(self, state):
        return RuntimeAction(action_type="retrieve_memory")


def test_runtime_stops_when_budget_is_exhausted():
    runtime = AgentRuntime(policy=NeverRespondPolicy())
    result = runtime.run(
        TurnContext(
            turn_id="turn-5",
            session_id="session-1",
            character_id="tsundere_sister",
            player_input="继续想",
            max_steps=2,
        )
    )

    assert result.stop_reason == "step_budget_exhausted"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/runtime/test_trace_sequence.py tests/runtime/test_stop_conditions.py -v`
Expected: FAIL until trace ordering and stop policy are fully wired

- [x] **Step 3: Write minimal implementation**

```python
# src/runtime/agent_runtime.py
class AgentRuntime:
    def _emit(self, trace, event_type, turn_id, step_index, **payload):
        trace.add_event(
            TurnEvent(
                event_type=event_type,
                turn_id=turn_id,
                step_index=step_index,
                payload=payload,
            )
        )
```

```python
# src/sandbox/tick_engine.py
class TickEngine:
    def _process_memory_decay(self, engine) -> dict:
        decay_results = {}

        for character_id, runtime in engine._pipelines.items():
            memory_adapter = getattr(runtime, "memory_adapter", None)
            if memory_adapter is None:
                continue
            decay_results[character_id] = {"status": "skipped_runtime_owned_memory"}

        return decay_results
```

```python
# src/sandbox/rumor_spreader.py
if get_memory_manager:
    mm = get_memory_manager(target_npc)
    if hasattr(mm, "consolidate"):
        mm.consolidate(
            player_input=f"[流言]{rumor.content}",
            npc_response="",
            emotion_score=3.0,
            importance=4.0,
            related_entities=[rumor.source_npc],
        )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/runtime/test_trace_sequence.py tests/runtime/test_stop_conditions.py tests/test_phase2.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/runtime/test_trace_sequence.py tests/runtime/test_stop_conditions.py tests/test_cognition.py tests/test_phase2.py src/runtime/agent_runtime.py src/sandbox/tick_engine.py src/sandbox/rumor_spreader.py
git commit -m "test: add trace-centric runtime regression coverage"
```
