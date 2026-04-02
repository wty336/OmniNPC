# OmniNPC Chroma Isolation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate recurring ChromaDB `disk I/O` / `readonly` failures by isolating the persistence boundary, adding deterministic preflight checks, and introducing controlled fallback behavior that keeps NPC chat available.

**Architecture:** Keep `MemoryManager` and higher-level cognition/runtime flows unchanged as much as possible. Move Chroma-specific logic behind a dedicated episodic store interface + factory, so `EpisodicMemory` only depends on stable contracts. Add health/degraded signals and operational tools for diagnosis and repair.

**Tech Stack:** Python 3.11, Pydantic v2, ChromaDB, pytest, Loguru

---

## Scope And Guardrails

**In scope**
- Chroma persistence isolation
- Chroma startup preflight and path validation
- Fail-safe fallback strategy for episodic memory
- Health visibility for memory backend
- Chroma repair/migration utility and runbook

**Out of scope**
- Runtime policy redesign
- Engine/API response schema changes
- Full cognition pipeline refactor
- Semantic memory/graph redesign

**Non-negotiable constraints**
- Keep current `MemoryManager.retrieve()` and `MemoryManager.consolidate()` call signatures unchanged
- Keep chat endpoint response shape unchanged
- Keep default gameplay path available even if Chroma is temporarily unavailable

---

## Planned File Map

**Create**
- `src/memory/stores/__init__.py`
- `src/memory/stores/base.py`
- `src/memory/stores/chroma_store.py`
- `src/memory/stores/in_memory_store.py`
- `src/memory/stores/factory.py`
- `src/memory/stores/health.py`
- `scripts/chroma/check_and_repair.py`
- `tests/memory/test_episodic_store_contracts.py`
- `tests/memory/test_chroma_store_preflight.py`
- `tests/memory/test_episodic_memory_fallback.py`
- `docs/superpowers/runbooks/chroma-ops.md`

**Modify**
- `config/settings.py`
- `.env.example`
- `src/memory/episodic_memory.py`
- `src/memory/memory_manager.py`
- `tests/config/test_settings.py`
- `tests/test_memory.py`

**Do not modify**
- `src/engine.py`
- `src/api/routes/chat.py`
- `src/cognition/*`
- `src/runtime/*`
- `src/adapters/llm/*`
- `src/adapters/tools/*`

---

### Task 1: Define Episodic Store Contracts And Settings

**Files:**
- Create: `src/memory/stores/base.py`
- Create: `src/memory/stores/health.py`
- Modify: `config/settings.py`
- Modify: `.env.example`
- Modify: `tests/config/test_settings.py`
- Test: `tests/memory/test_episodic_store_contracts.py`

- [ ] **Step 1: Write failing tests for contract and settings**

```python
def test_memory_backend_defaults():
    assert settings.memory.episodic_backend == "chroma"
    assert settings.memory.episodic_on_error == "fallback_in_memory"
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_episodic_store_contracts.py tests/config/test_settings.py -v
```

Expected: FAIL due to missing settings/contract.

- [ ] **Step 3: Add settings and base contracts**

```python
class EpisodicStore(Protocol):
    def store(self, memory: MemoryItem) -> None: ...
    def retrieve(self, query: str, top_k: int, min_importance: float = 0.0) -> list[MemoryItem]: ...
    def count(self) -> int: ...
    def health(self) -> MemoryBackendHealth: ...
```

- [ ] **Step 4: Re-run tests**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_episodic_store_contracts.py tests/config/test_settings.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memory/stores/base.py src/memory/stores/health.py config/settings.py .env.example tests/memory/test_episodic_store_contracts.py tests/config/test_settings.py
git commit -m "feat(memory): add episodic store contracts and backend settings"
```

### Task 2: Implement Chroma Preflight And Path Isolation

**Files:**
- Create: `src/memory/stores/chroma_store.py`
- Modify: `tests/memory/test_chroma_store_preflight.py` (create if missing)
- Modify: `config/settings.py` (only if Task 1 lacks needed knobs)

- [ ] **Step 1: Write failing preflight tests**

```python
def test_chroma_store_reports_degraded_when_path_not_writable(monkeypatch, tmp_path):
    ...
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_chroma_store_preflight.py -v
```

Expected: FAIL because `ChromaEpisodicStore` does not exist.

- [ ] **Step 3: Implement deterministic preflight**

```python
class ChromaEpisodicStore(EpisodicStore):
    # preflight:
    # 1) ensure path exists
    # 2) check read/write temp marker
    # 3) initialize PersistentClient
    # 4) map known exceptions to health status + reason
```

- [ ] **Step 4: Re-run preflight tests**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_chroma_store_preflight.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memory/stores/chroma_store.py tests/memory/test_chroma_store_preflight.py
git commit -m "feat(memory): add chroma preflight and health reporting"
```

### Task 3: Add Controlled Fallback Store And Factory

**Files:**
- Create: `src/memory/stores/in_memory_store.py`
- Create: `src/memory/stores/factory.py`
- Test: `tests/memory/test_episodic_memory_fallback.py`

- [ ] **Step 1: Write failing fallback tests**

```python
def test_factory_falls_back_to_in_memory_on_chroma_error():
    ...
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_episodic_memory_fallback.py -v
```

Expected: FAIL because factory/fallback is missing.

- [ ] **Step 3: Implement fallback policy**

```python
def build_episodic_store(character_id: str, settings: MemorySettings) -> EpisodicStore:
    # backend=chroma -> try chroma
    # on_error=fallback_in_memory -> return in-memory store + degraded health
    # on_error=raise -> re-raise startup/runtime error
```

- [ ] **Step 4: Re-run tests**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_episodic_memory_fallback.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memory/stores/in_memory_store.py src/memory/stores/factory.py tests/memory/test_episodic_memory_fallback.py
git commit -m "feat(memory): add episodic store factory and fallback strategy"
```

### Task 4: Refactor EpisodicMemory To Store Adapter

**Files:**
- Modify: `src/memory/episodic_memory.py`
- Modify: `src/memory/memory_manager.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write failing integration tests for adapterized EpisodicMemory**

```python
def test_episodic_memory_uses_store_contract():
    ...
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/test_memory.py tests/memory/test_episodic_store_contracts.py -v
```

Expected: FAIL because `EpisodicMemory` still directly instantiates `PersistentClient`.

- [ ] **Step 3: Refactor with minimal surface change**

```python
class EpisodicMemory:
    def __init__(self, character_id: str, store: EpisodicStore | None = None):
        self._store = store or build_episodic_store(character_id, settings.memory)
```

- [ ] **Step 4: Re-run tests**

Run:

```powershell
conda activate omni_npc; python -m pytest tests/test_memory.py tests/memory/test_episodic_store_contracts.py tests/memory/test_episodic_memory_fallback.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memory/episodic_memory.py src/memory/memory_manager.py tests/test_memory.py
git commit -m "refactor(memory): isolate episodic memory behind store adapter"
```

### Task 5: Add Operational Diagnostics And Repair Script

**Files:**
- Create: `scripts/chroma/check_and_repair.py`
- Create: `docs/superpowers/runbooks/chroma-ops.md`
- Modify: `src/memory/episodic_memory.py` (only for exposing health snapshot if needed)

- [ ] **Step 1: Add script-level checks**

Script responsibilities:
- verify configured persist dir exists and is writable
- run minimal client init + collection open
- print health report JSON
- optional `--repair` to backup and rotate corrupted dir

- [ ] **Step 2: Validate script in dry-run**

Run:

```powershell
conda activate omni_npc; python scripts/chroma/check_and_repair.py --dry-run
```

Expected: Exit 0 with structured health output.

- [ ] **Step 3: Document runbook**

Runbook must include:
- symptom matrix (`disk I/O`, `readonly`, startup init failure)
- triage checklist
- safe repair steps
- rollback steps

- [ ] **Step 4: Commit**

```bash
git add scripts/chroma/check_and_repair.py docs/superpowers/runbooks/chroma-ops.md src/memory/episodic_memory.py
git commit -m "docs(ops): add chroma diagnostics and repair runbook"
```

### Task 6: Full Regression And Rollout Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-04-02-omni-npc-chroma-isolation-hardening.md` (status updates)
- Test only: no required `src/*` changes

- [ ] **Step 1: Run focused memory regression**

```powershell
conda activate omni_npc; python -m pytest tests/memory/test_episodic_store_contracts.py tests/memory/test_chroma_store_preflight.py tests/memory/test_episodic_memory_fallback.py tests/test_memory.py -v
```

- [ ] **Step 2: Run runtime/API smoke regression**

```powershell
conda activate omni_npc; python -m pytest tests/runtime/test_engine_runtime_integration.py tests/runtime/test_runtime_legacy_parity.py tests/test_api.py -v
```

Expected:
- memory suite PASS
- runtime parity PASS
- API tests PASS or explicit SKIP if dependency missing

- [ ] **Step 3: Manual smoke on local backend**

Run:

```powershell
conda activate omni_npc; python -m uvicorn src.api.main:app --reload
```

Then hit one chat request and confirm:
- response not empty
- no Chroma uncaught exception in logs
- if fallback occurred, degraded reason is logged once with clear context

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-04-02-omni-npc-chroma-isolation-hardening.md
git commit -m "chore(plan): mark chroma hardening rollout verification"
```

---

## Rollout Strategy

1. Keep `episodic_backend=chroma` as default.
2. Set `episodic_on_error=fallback_in_memory` in non-critical environments first.
3. Observe degraded/fallback rate for at least one test cycle.
4. For production-like environments, decide between:
- `fallback_in_memory` (higher availability, weaker persistence)
- `raise` (strict persistence guarantee, may fail fast)

## Risk Register

- **Risk:** silent fallback hides persistence loss  
  **Mitigation:** explicit degraded health + one-shot warning + runbook alert rule.

- **Risk:** path/permission behavior differs across Windows machines  
  **Mitigation:** deterministic preflight and a standalone repair utility.

- **Risk:** migration from old corrupted Chroma dir loses data  
  **Mitigation:** backup-before-repair and documented restore procedure.

## Done Criteria

- Chroma failures no longer crash chat flow by default
- Fallback behavior is deterministic and test-covered
- Health/degraded state is queryable and logged
- Repair path is documented and script-backed
- Existing runtime/chat API contracts remain unchanged

