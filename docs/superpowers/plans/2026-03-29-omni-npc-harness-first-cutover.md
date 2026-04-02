# OmniNPC Harness-First Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote OmniNPC from a dual-path runtime skeleton to a harness-first chat engine where `AgentRuntime` becomes the default execution path without changing the external chat response shape.

**Architecture:** Keep the current adapter boundaries and dual-path safety net, but move decision ownership into runtime. The new work centers on a richer runtime policy, a side-effect-free action planning ability, parity tests between runtime and legacy behavior, and a controlled default-path flip that still preserves a fallback route.

**Tech Stack:** Python 3.10, FastAPI, Pydantic v2, Loguru, pytest

---

## Planned File Map

**Create**
- `src/runtime/chat_policy.py` — runtime policy that sequences memory retrieval, reflection, tool proposal, and response generation using the already-created adapters.
- `src/cognition/action_planner.py` — side-effect-free planning helper that can propose dialogue and tool calls without executing tools.
- `tests/runtime/test_chat_runtime_policy.py` — unit tests for the richer runtime policy.
- `tests/cognition/test_action_planner.py` — unit tests for the side-effect-free action planner.
- `tests/runtime/test_runtime_legacy_parity.py` — comparison tests between legacy pipeline output shape and runtime output shape.

**Modify**
- `src/runtime/agent_runtime.py` — allow runtime state to carry injected dependencies and richer intermediate artifacts.
- `src/runtime/policies.py` — keep `DefaultRuntimePolicy` as a smoke-test path, export the new chat policy separately or from this package.
- `src/cognition/action_generator.py` — keep the legacy compatibility wrapper, but delegate planning to the new planner so tool execution can move out of cognition for runtime-enabled paths.
- `src/engine.py` — build runtime with real dependencies and flip the default constructor path only after parity tests pass.
- `src/api/routes/chat.py` — no behavioral change expected; only verify compatibility after the default path flip.
- `tests/runtime/test_engine_runtime_integration.py` — extend integration coverage for the richer runtime path.
- `tests/test_api.py` — only if needed to keep current response compatibility assertions valid once runtime becomes default.

**Leave Alone**
- `src/cognition/pipeline.py`
- `src/cognition/perception.py`
- `src/cognition/inner_monologue.py`
- `src/adapters/llm/*`
- `src/adapters/tools/*`
- `src/adapters/memory/*`
- `src/models/message.py`

---

### Task 8: Build A Real Chat Runtime Policy

**Files:**
- Create: `src/runtime/chat_policy.py`
- Modify: `src/runtime/agent_runtime.py`
- Modify: `src/runtime/policies.py`
- Test: `tests/runtime/test_chat_runtime_policy.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_chat_runtime_policy.py -v`
- Result: `4 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_agent_runtime.py tests/runtime/test_chat_runtime_policy.py -v`
- Result: `7 passed`

**Actual implementation notes discovered during Task 8 review:**
- `ChatRuntimePolicy` now drives the minimal runtime action order: `retrieve_memory -> reflect -> plan_response -> propose_tool -> respond`.
- `AgentRuntime` now preserves intermediate state for `memory_snapshot`, `reflection`, and `planned_response` without changing `RuntimeResult` or adding event types.
- `AgentRuntime` now accepts optional collaborators for the chat runtime path and fails early with a clear configuration error if `ChatRuntimePolicy` is used without the required collaborators.
- `src.runtime.policies` re-exports `ChatRuntimePolicy`, and that import surface is covered by tests.

- [x] **Step 1: Write the failing policy tests**

```python
from src.runtime.actions import RuntimeAction
from src.runtime.chat_policy import ChatRuntimePolicy


def test_chat_runtime_policy_retrieves_memory_first():
    policy = ChatRuntimePolicy()
    action = policy.next_action(
        {
            "memory_snapshot": None,
            "reflection": None,
            "planned_response": None,
        }
    )
    assert action == RuntimeAction(action_type="retrieve_memory")


def test_chat_runtime_policy_reflects_after_memory():
    policy = ChatRuntimePolicy()
    action = policy.next_action(
        {
            "memory_snapshot": {"episodic_memories": []},
            "reflection": None,
            "planned_response": None,
        }
    )
    assert action == RuntimeAction(action_type="reflect")


def test_chat_runtime_policy_plans_before_responding():
    policy = ChatRuntimePolicy()
    action = policy.next_action(
        {
            "memory_snapshot": {"episodic_memories": []},
            "reflection": "她需要先安抚玩家。",
            "planned_response": None,
        }
    )
    assert action == RuntimeAction(action_type="plan_response")


def test_chat_runtime_policy_responds_after_plan():
    policy = ChatRuntimePolicy()
    action = policy.next_action(
        {
            "memory_snapshot": {"episodic_memories": []},
            "reflection": "她需要先安抚玩家。",
            "planned_response": {"dialogue": "先别慌，我来处理。"},
        }
    )
    assert action == RuntimeAction(
        action_type="respond",
        payload={"dialogue": "先别慌，我来处理。"},
    )
```

- [x] **Step 2: Run the new policy test to verify it fails**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_chat_runtime_policy.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `src.runtime.chat_policy` or missing `ChatRuntimePolicy`.

- [x] **Step 3: Implement the minimal richer policy**

```python
# src/runtime/chat_policy.py
from __future__ import annotations

from src.runtime.actions import RuntimeAction


class ChatRuntimePolicy:
    def next_action(self, state: dict) -> RuntimeAction:
        if state.get("memory_snapshot") is None:
            return RuntimeAction(action_type="retrieve_memory")
        if state.get("reflection") is None:
            return RuntimeAction(action_type="reflect")
        if state.get("planned_response") is None:
            return RuntimeAction(action_type="plan_response")

        planned = state.get("planned_response") or {}
        tool_name = planned.get("tool_name")
        if tool_name and not state.get("last_tool_execution"):
            return RuntimeAction(
                action_type="propose_tool",
                payload={
                    "tool_name": tool_name,
                    "arguments": planned.get("arguments", {}),
                },
            )

        return RuntimeAction(
            action_type="respond",
            payload={"dialogue": planned.get("dialogue", "……")},
        )
```

```python
# src/runtime/policies.py
from src.runtime.chat_policy import ChatRuntimePolicy

__all__ = ["RuntimePolicy", "DefaultRuntimePolicy", "ChatRuntimePolicy"]
```

- [x] **Step 4: Teach AgentRuntime to preserve intermediate state without new side effects**

```python
# src/runtime/agent_runtime.py
if action.action_type == "retrieve_memory":
    memory_adapter = state["memory_adapter"]
    state["memory_snapshot"] = memory_adapter.retrieve(
        query=context.player_input,
        player_id=state["game_state"].player.player_id,
    ).model_dump()
elif action.action_type == "reflect":
    reflector = state["reflector"]
    state["reflection"] = reflector.generate(
        context=state["perception_context"],
    )
elif action.action_type == "plan_response":
    planner = state["action_planner"]
    plan = planner.plan(
        context=state["perception_context"],
        inner_monologue=state["reflection"],
    )
    state["planned_response"] = plan
```

Keep the implementation minimal:
- do not add new event types yet
- do not change `RuntimeResult`
- keep `STEP_COMPLETED` as the completion event for every step

- [x] **Step 5: Run the policy tests to verify they pass**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_chat_runtime_policy.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

Deferred by user request: Task 8 remains uncommitted for now.

```bash
git add src/runtime/chat_policy.py src/runtime/agent_runtime.py src/runtime/policies.py tests/runtime/test_chat_runtime_policy.py
git commit -m "feat: add chat runtime policy"
```

### Task 9: Split Action Planning From Tool Execution

**Files:**
- Create: `src/cognition/action_planner.py`
- Modify: `src/cognition/action_generator.py`
- Test: `tests/cognition/test_action_planner.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/cognition/test_action_planner.py -v`
- Result: `2 passed`
- Regression check: `conda activate plante; python -m pytest tests/cognition/test_action_planner.py tests/cognition/test_reflection_and_action_policy.py -v`
- Result: `5 passed`

**Actual implementation notes discovered during Task 9 review:**
- `src/cognition/action_planner.py` was added as a side-effect-free planning layer that uses `model_adapter.complete(ModelRequest(...))`.
- `ActionGenerator` now delegates the first planning call to `ActionPlanner`, but still owns tool execution and the legacy second LLM call.
- To preserve legacy multi-tool compatibility, the planner keeps an internal ordered `tool_calls` carrier while still exposing `dialogue`, `tool_name`, and `arguments` from the first proposal.
- Task 9 tests were tightened after review:
  - planner side-effect freedom is now checked via unchanged `game_state`
  - single-tool and multi-tool legacy paths are tested separately

- [x] **Step 1: Write the failing planner tests**

```python
from src.cognition.action_planner import ActionPlanner


class FakeContext:
    def to_messages(self, inner_monologue: str):
        return [
            {"role": "system", "content": inner_monologue},
            {"role": "user", "content": "师姐救我"},
        ]


class FakeModelAdapter:
    def complete(self, request):
        return type(
            "Response",
            (),
            {
                "content": "先别慌，我来处理。",
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "function": {
                            "name": "update_affection",
                            "arguments": "{\"source_id\": \"npc\", \"target_id\": \"player\", \"delta\": 5}",
                        },
                    }
                ],
            },
        )()


def test_action_planner_returns_tool_proposal_without_executing():
    planner = ActionPlanner(model_adapter=FakeModelAdapter())
    result = planner.plan(context=FakeContext(), inner_monologue="先安抚她。")

    assert result["dialogue"] == "先别慌，我来处理。"
    assert result["tool_name"] == "update_affection"
    assert result["arguments"]["delta"] == 5
```

- [x] **Step 2: Run the planner test to verify it fails**

Run:

```powershell
conda activate plante; python -m pytest tests/cognition/test_action_planner.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `src.cognition.action_planner`.

- [x] **Step 3: Implement the side-effect-free planner**

```python
# src/cognition/action_planner.py
from __future__ import annotations

import json

from src.adapters.llm.ark_adapter import ArkModelAdapter
from src.adapters.llm.base import ModelRequest
from src.tools.base import get_tool_definitions


class ActionPlanner:
    def __init__(self, model_adapter: ArkModelAdapter | None = None):
        self._model = model_adapter or ArkModelAdapter()

    def plan(self, context, inner_monologue: str) -> dict:
        messages = [
            {"role": "system", "content": inner_monologue},
            {"role": "user", "content": context.player_input},
        ]
        result = self._model.complete(
            ModelRequest(
                purpose="respond",
                messages=messages,
                tools=get_tool_definitions(),
                temperature=0.8,
            )
        )

        plan = {
            "dialogue": result.content or "",
            "tool_name": None,
            "arguments": {},
        }
        if result.tool_calls:
            tool_call = result.tool_calls[0]
            plan["tool_name"] = tool_call["function"]["name"]
            plan["arguments"] = json.loads(tool_call["function"]["arguments"])
        return plan
```

- [x] **Step 4: Keep the legacy wrapper but move execution back into the wrapper only**

```python
# src/cognition/action_generator.py
from src.cognition.action_planner import ActionPlanner


class ActionGenerator:
    def __init__(self, model_adapter: ArkModelAdapter | None = None):
        self._planner = ActionPlanner(model_adapter=model_adapter)

    def generate(self, context, inner_monologue: str) -> AgentResponse:
        plan = self._planner.plan(context=context, inner_monologue=inner_monologue)
        dialogue = plan["dialogue"]
        tool_call_results = []

        if plan["tool_name"]:
            tool_registry = get_tool_registry()
            tool_func = tool_registry.get(plan["tool_name"])
            if tool_func is not None:
                result = tool_func(game_state=context.game_state, **plan["arguments"])
                tool_call_results.append(
                    ToolCallResult(
                        tool_name=plan["tool_name"],
                        arguments=plan["arguments"],
                        result=result,
                        success=True,
                    )
                )

        return AgentResponse(
            dialogue=dialogue,
            inner_monologue=inner_monologue,
            tool_calls=tool_call_results,
            character_id=context.character.id,
            character_name=context.character.name,
        )
```

Requirement for this step:
- runtime-enabled path must be able to use `ActionPlanner` directly
- legacy pipeline must still keep its old external behavior
- do not delete the legacy second LLM call unless tests prove it is no longer needed

- [x] **Step 5: Run planner tests to verify they pass**

Run:

```powershell
conda activate plante; python -m pytest tests/cognition/test_action_planner.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

Deferred by user request: Task 9 remains uncommitted for now.

```bash
git add src/cognition/action_planner.py src/cognition/action_generator.py tests/cognition/test_action_planner.py
git commit -m "refactor: split action planning from tool execution"
```

### Task 10: Wire Runtime To Real Engine Dependencies

**Files:**
- Modify: `src/runtime/agent_runtime.py`
- Modify: `src/engine.py`
- Modify: `tests/runtime/test_engine_runtime_integration.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py -v`
- Result: `5 passed`
- Regression check: `conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py tests/runtime/test_chat_runtime_policy.py tests/cognition/test_action_planner.py tests/cognition/test_reflection_and_action_policy.py -v`
- Result: `14 passed`

**Actual implementation notes discovered during Task 10 review:**
- `NPCEngine._get_or_create_runtime()` now builds a real runtime path with:
  - `ChatRuntimePolicy`
  - `ToolExecutor(ToolCatalog.load())`
  - `CompositeMemoryAdapter`
  - `Perception(memory_adapter=...)`
  - `InnerMonologue()`
  - `ActionPlanner()`
  - loaded `character`
- `process_chat()` now passes the current persisted `game_state` into `runtime.run(...)`.
- `AgentRuntime` now supports lazy per-turn `perception_context` construction when a real `Perception` and `character` are supplied.
- During quality review, legacy pipeline creation was moved behind a lazy `_get_or_create_pipeline()` factory so the runtime path no longer depends on importing the legacy pipeline stack.
- `tests/runtime/test_engine_runtime_integration.py` now restores the `sys.modules` entries it mutates, so the runtime import tests stay self-contained and order-independent.

- [x] **Step 1: Write the failing integration test for real dependency injection**

```python
from src.engine import NPCEngine


def test_engine_runtime_path_builds_runtime_with_real_dependencies(tmp_path):
    engine = NPCEngine(use_agent_runtime=True)
    engine.load_character("tsundere_sister", yaml_path="tests/fixtures/tsundere_sister.yaml")

    runtime = engine._get_or_create_runtime("tsundere_sister")

    assert runtime.tool_executor is not None
    assert runtime.policy.__class__.__name__ == "ChatRuntimePolicy"
```

- [x] **Step 2: Run the integration test to verify it fails**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py -v
```

Expected: FAIL because the runtime still uses `DefaultRuntimePolicy()` and does not inject real adapters.

- [x] **Step 3: Build runtime with actual memory/model/tool dependencies**

```python
# src/engine.py
from src.adapters.memory.composite import CompositeMemoryAdapter
from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.executor import ToolExecutor
from src.cognition.action_planner import ActionPlanner
from src.cognition.inner_monologue import InnerMonologue
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.chat_policy import ChatRuntimePolicy


def _get_or_create_runtime(self, character_id: str) -> AgentRuntime:
    runtime = self._runtimes.get(character_id)
    if runtime is None:
        memory_manager = self._memory_managers[character_id]
        runtime = AgentRuntime(
            policy=ChatRuntimePolicy(),
            tool_executor=ToolExecutor(ToolCatalog.load()),
            memory_adapter=CompositeMemoryAdapter(memory_manager),
            reflector=InnerMonologue(),
            action_planner=ActionPlanner(),
        )
        self._runtimes[character_id] = runtime
    return runtime
```

```python
# src/runtime/agent_runtime.py
class AgentRuntime:
    def __init__(
        self,
        policy,
        tool_executor=None,
        memory_adapter=None,
        reflector=None,
        action_planner=None,
    ):
        self.policy = policy
        self.tool_executor = tool_executor
        self.memory_adapter = memory_adapter
        self.reflector = reflector
        self.action_planner = action_planner
```

- [x] **Step 4: Preserve the current API response shape while enriching runtime state**

```python
# src/engine.py
runtime_result = runtime.run(turn_context)
response = self._adapt_runtime_result(character, runtime_result)

assert set(response.model_dump().keys()) == {
    "dialogue",
    "emotion",
    "inner_monologue",
    "tool_calls",
    "state_changes",
    "metadata",
    "character_id",
    "character_name",
}
```

This step is complete only if:
- `process_chat()` stays synchronous
- `ChatResponse` shape stays unchanged
- legacy pipeline branch still works untouched

- [x] **Step 5: Run the engine integration test to verify it passes**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

Deferred by user request: Task 10 remains uncommitted for now.

```bash
git add src/runtime/agent_runtime.py src/engine.py tests/runtime/test_engine_runtime_integration.py
git commit -m "feat: inject real dependencies into runtime path"
```

### Task 11: Add Runtime vs Legacy Parity Tests And Restore API Test Preconditions

**Files:**
- Create: `tests/runtime/test_runtime_legacy_parity.py`
- Modify: `tests/test_api.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py -v`
- Result: `1 passed`
- `conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py tests/test_api.py -v`
- Result: `1 passed, 1 skipped`

**Actual implementation notes discovered during Task 11 review:**
- `tests/runtime/test_runtime_legacy_parity.py` now verifies response-shape parity between legacy and runtime paths at the `AgentResponse` contract level, not full real-dependency behavioral parity.
- The parity test uses local test doubles and import stubs for `loguru`, `config.settings`, `GameStateStore`, and sandbox modules so the test stays isolated from the current `plante` environment gaps.
- `tests/test_api.py` now declares `fastapi` as an explicit environment precondition via `pytest.importorskip("fastapi")`, so the suite skips cleanly instead of failing during collection.
- No `src/*` files were changed in Task 11; the task remained fully within test and plan-document scope.

- [x] **Step 1: Write the failing parity tests**

```python
from src.engine import NPCEngine


def test_runtime_and_legacy_paths_return_same_response_shape():
    legacy = NPCEngine(use_agent_runtime=False)
    runtime = NPCEngine(use_agent_runtime=True)

    legacy_response = legacy.process_chat("师姐救我", "tsundere_sister", "s1")
    runtime_response = runtime.process_chat("师姐救我", "tsundere_sister", "s1")

    assert set(legacy_response.model_dump().keys()) == set(runtime_response.model_dump().keys())
    assert runtime_response.dialogue
```

- [x] **Step 2: Run the parity test to verify it fails**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py -v
```

Expected: FAIL because runtime path still lacks parity on one or more output fields.

- [x] **Step 3: Restore API test preconditions inside `plante` without changing the API contract**

```python
# tests/test_api.py
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
```

This step is intentionally minimal:
- do not install dependencies from the plan
- do not change route behavior
- make the API test suite explicit about the current environment precondition

- [x] **Step 4: Tighten parity assertions to what actually matters**

```python
assert runtime_response.character_id == legacy_response.character_id
assert runtime_response.character_name == legacy_response.character_name
assert isinstance(runtime_response.dialogue, str)
assert runtime_response.tool_calls is not None
assert runtime_response.state_changes is not None
```

- [x] **Step 5: Run parity and API tests**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py tests/test_api.py -v
```

Expected:
- parity tests PASS
- API tests either PASS or SKIP explicitly because `fastapi` is unavailable

- [ ] **Step 6: Commit**

Deferred by user request: Task 11 remains uncommitted for now.

```bash
git add tests/runtime/test_runtime_legacy_parity.py tests/test_api.py
git commit -m "test: add runtime parity coverage and clarify api test preconditions"
```

### Task 12: Flip The Default Chat Path To Runtime With Legacy Fallback

**Files:**
- Modify: `src/engine.py`
- Modify: `tests/runtime/test_engine_runtime_integration.py`
- Modify: `tests/runtime/test_runtime_legacy_parity.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py -v`
- Result: `6 passed`
- `conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py -v`
- Result: `1 passed`

**Actual implementation notes discovered during Task 12 review:**
- The only production-code change was in `get_engine()`, which now constructs the global singleton with `NPCEngine(use_agent_runtime=True)`.
- `NPCEngine(use_agent_runtime=False)` remains unchanged, so the explicit legacy fallback path is still available.
- `tests/runtime/test_engine_runtime_integration.py` now includes coverage that resets the global `_engine` singleton and verifies the default global engine path is runtime-backed.
- No API contract changes were introduced, and no runtime/cognition internals were expanded in this task.

- [x] **Step 1: Write the failing default-path test**

```python
from src.engine import get_engine


def test_global_engine_defaults_to_runtime(monkeypatch):
    from src import engine as engine_module

    engine_module._engine = None
    engine = get_engine()

    assert engine._use_agent_runtime is True
```

- [x] **Step 2: Run the default-path test to verify it fails**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py -v
```

Expected: FAIL because `get_engine()` still constructs `NPCEngine()` with the legacy-default flag.

- [x] **Step 3: Flip the global default but keep constructor-level fallback**

```python
# src/engine.py
def get_engine() -> NPCEngine:
    global _engine
    if _engine is None:
        _engine = NPCEngine(use_agent_runtime=True)
    return _engine
```

Keep this guarantee:
- callers can still opt into `NPCEngine(use_agent_runtime=False)` explicitly
- no global settings flag is introduced

- [x] **Step 4: Re-run integration and parity coverage**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_engine_runtime_integration.py tests/runtime/test_runtime_legacy_parity.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

Deferred by user request: Task 12 remains uncommitted for now.

```bash
git add src/engine.py tests/runtime/test_engine_runtime_integration.py tests/runtime/test_runtime_legacy_parity.py
git commit -m "feat: make runtime the default engine path"
```

### Task 13: Quarantine The Legacy Pipeline

**Files:**
- Modify: `src/engine.py`
- Modify: `src/cognition/action_generator.py`
- Test: `tests/runtime/test_runtime_legacy_parity.py`

**Status:** Completed, not committed

**Actual verification result:**
- `conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py -v`
- Result: `2 passed`
- `conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py tests/runtime/test_engine_runtime_integration.py -v`
- Result: `8 passed`

**Actual implementation notes discovered during Task 13 review:**
- `tests/runtime/test_runtime_legacy_parity.py` now includes explicit fallback coverage for `NPCEngine(use_agent_runtime=False)`.
- `src/engine.py` now documents the runtime branch as the default chat path and the legacy branch as compatibility fallback only.
- `src/cognition/action_generator.py` is now explicitly documented as a legacy compatibility wrapper; runtime-enabled paths are expected to use `ActionPlanner + ToolExecutor`.
- Task 13 did not delete or rewrite legacy behavior. It only tightened tests and clarified boundaries in code.

- [x] **Step 1: Write the failing quarantine test**

```python
from src.engine import NPCEngine


def test_legacy_pipeline_remains_available_as_explicit_fallback():
    engine = NPCEngine(use_agent_runtime=False)
    response = engine.process_chat("师姐救我", "tsundere_sister", "s1")

    assert response.dialogue
```

- [x] **Step 2: Run the quarantine test to verify it fails if fallback is broken**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py -v
```

Expected: PASS today; keep this step in the plan to protect the explicit fallback before further cleanup.

- [x] **Step 3: Make the legacy-only boundaries explicit in code comments and branch names**

```python
# src/engine.py
if self._use_agent_runtime:
    runtime = self._get_or_create_runtime(character_id)
    runtime_result = runtime.run(turn_context)
    response = self._adapt_runtime_result(character, runtime_result)
else:
    # Legacy compatibility path. Keep until runtime path fully replaces pipeline behavior.
    response = pipeline.run(
        player_input=player_input,
        character=character,
        game_state=game_state,
    )
```

```python
# src/cognition/action_generator.py
# Legacy compatibility wrapper. Runtime-enabled paths should use ActionPlanner + ToolExecutor.
class ActionGenerator:
    def __init__(self, model_adapter: ArkModelAdapter | None = None):
        self._planner = ActionPlanner(model_adapter=model_adapter)
```

- [x] **Step 4: Run the focused quarantine regression**

Run:

```powershell
conda activate plante; python -m pytest tests/runtime/test_runtime_legacy_parity.py tests/runtime/test_engine_runtime_integration.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

Deferred by user request: Task 13 remains uncommitted for now.

```bash
git add src/engine.py src/cognition/action_generator.py tests/runtime/test_runtime_legacy_parity.py tests/runtime/test_engine_runtime_integration.py
git commit -m "refactor: quarantine legacy pipeline as fallback path"
```

---

## Cross-Task Verification

Run after each completed task batch:

```powershell
conda activate plante; python -m pytest tests/runtime/test_runtime_contracts.py tests/runtime/test_agent_runtime.py tests/runtime/test_trace_sequence.py tests/runtime/test_stop_conditions.py tests/runtime/test_chat_runtime_policy.py tests/runtime/test_runtime_tool_flow.py tests/runtime/test_engine_runtime_integration.py tests/runtime/test_runtime_legacy_parity.py tests/adapters/test_model_adapter.py tests/adapters/test_tool_executor.py tests/adapters/test_memory_adapter.py tests/cognition/test_perception_ability.py tests/cognition/test_reflection_and_action_policy.py tests/cognition/test_action_planner.py tests/test_cognition.py tests/test_phase2.py tests/test_api.py -v
```

Expected:
- runtime tests PASS
- adapter tests PASS
- cognition tests PASS
- API tests PASS or explicit SKIP when `fastapi` is unavailable in `plante`

## Guardrails

- Do not change the external `ChatResponse` shape.
- Do not reintroduce a global runtime flag in `config.settings`.
- Do not delete `CognitivePipeline` during this phase.
- Do not let runtime-enabled paths execute tools inside cognition modules.
- Keep all commands on the `plante` conda environment via `conda activate plante; python -m pytest ...`.
