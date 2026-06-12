# Context Engineering — Implementation Plan

**Architecture (1:1):** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**ADR:** [`ADR-CTX-001`](../adr/ADR-CTX-001.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Status summary (2026-06-12)

| Phase | Scope | Status |
|-------|-------|--------|
| **CTX** (control plane closeout) | `context_runtime_bridge`, `context_wiring`, Appendix L | **Done** (2026-06-02) |
| **R-Context** | Budget API, `CONTEXT_*` events | **Done** |
| **MEM-DEPTH-1.\*** | `ContextCompiler`, `DegradationLadder`, preflight | **Done** (owned by MEMORY delivery; CE consumes) |
| **CE-DOC** | Domain split + architecture + plan | **Done** (2026-06-12) |
| **CE-EXT** | Plugin engine + step-aware + codebase preset | **Planned** — this register |

**As-built maturity:** L2.5 engine / L3 control plane — see architecture §3.

**Delivery rule:** One **CE-\*** ID per PR → update master table + gap register → `pytest -m gate` + domain CI scripts green.

---

## Gap traceability matrix (GAP-CTX → CE)

| GAP-CTX | CE IDs | Wave |
|---------|--------|------|
| GAP-CTX-01 | CE-1, CE-2 | 1 |
| GAP-CTX-02 | CE-3 | 2 |
| GAP-CTX-03 | CE-3.4 | 2 |
| GAP-CTX-04 | CE-4 | 3 |
| GAP-CTX-05 | CE-10 | 5 |
| GAP-CTX-06 | CE-7 | 4 |
| GAP-CTX-07 | CE-8 | 4 |
| GAP-CTX-08 | CE-10.3 | 5 |
| GAP-CTX-09 | CE-9.2 | 5 |
| GAP-CTX-10 | CE-9.1 | 5 |
| GAP-CTX-11 | CE-3.6 | 2 |

---

## Phase CE-DOC — Domain documentation split

**Status:** **Done** (2026-06-12)

| ID | Deliverable | Status |
|----|-------------|--------|
| CE-DOC.1 | `architecture/CONTEXT_ENGINEERING.md` — full canon | **Done** |
| CE-DOC.2 | `plan/CONTEXT_ENGINEERING.md` — this register | **Done** |
| CE-DOC.3 | `ADR-CTX-001` domain split | **Done** |
| CE-DOC.4 | Hub + audit map + `guides/audit/CONTEXT_ENGINEERING.md` | **Done** |
| CE-DOC.5 | MEMORY canon cross-links (Layer C → CE) | **Done** |
| CE-DOC.6 | `generate_domain_audit_prompts.py` MEMORY/CE split | **Done** |

---

## Phase CE-EXT — Context Engineering Plugin Engine

**Status:** **Planned**  
**Goal:** Close all **GAP-CTX-\*** rows — production-grade plugin engine integrated with Harness observability, policy, and Tier-3 presets.  
**Prerequisites:** CE-DOC Done · MEM-DEPTH-1 Done · CTX Done · OBS event spine Done  
**Success gate:** L3+ engine on FAUDIT layer 16 · reference `codebase` preset on lab host · gate green

### Execution order (waves)

```text
Wave 1 (P0): CE-1 contracts + CE-2 registry
Wave 2 (P0): CE-3 DefaultNexusContextEngine unification
Wave 3 (P1): CE-4 step-aware assembly
Wave 4 (P1): CE-7 workspace provider + CE-8 orchestrator (optional hop)
Wave 5 (P2): CE-9 observability + CE-10 quality in hot path
Wave 6 (P2): CE-11 Tier-3 presets + CE-12 DX / scaffold / gates
```

---

### Wave 1 — Contracts and plugin registry (P0)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-1.1** | Create `intergrax/context/contracts.py` — `ContextFragment`, `ContextFragmentSource`, `AssembledContext`, `BudgetAllocationResult` | P0 | Planned | Unit tests for frozen dataclasses / enums |
| **CE-1.2** | `ContextAssemblyRequest` + `ContextProviderContext` (runtime handles via typed ctx object) | P0 | Planned | Schema version field; no secrets in repr |
| **CE-1.3** | Protocols: `ContextSourceProvider`, `ContextRanker`, `ContextBudgetAllocator`, `ContextFormatter`, `ContextValidator`, `ContextEngine` | P0 | Planned | `typing.Protocol` + docstrings; gate import boundary test |
| **CE-1.4** | `ContextPlugin` dataclass + `ContextPluginRegistry` | P0 | Planned | Register/list/unregister providers |
| **CE-1.5** | Move shared scoring types from `context_engineering.py` → `intergrax/context/quality.py` (re-export shim) | P1 | Planned | No breaking imports in one release — deprecation event |
| **CE-1.6** | Architecture gate: `intergrax/context/` MUST NOT import `agents/` or `applications/` | P0 | Planned | `check_intergrax_no_applications_imports` pattern script or extend existing |
| **CE-2.1** | `register_context_plugin()` + `intergrax.context` entry point group in `pyproject.toml` | P0 | Planned | Third-party plugin discoverable |
| **CE-2.2** | `bootstrap_context_catalog()` in `intergrax/core/catalog_bootstrap.py` | P0 | Planned | Called from `wire_application_environment` behind flag |
| **CE-2.3** | Shipped `BuiltinContextPlugin` registering all §8.4 providers (stubs delegating to as-built steps) | P0 | Planned | Integration test: registry lists ≥10 providers |
| **CE-2.4** | `ContextProfile.context_plugin_ids` + validation against registry | P1 | Planned | Unknown id → fail at wire time (lab) / warn (prod) |
| **CE-2.5** | Unit tests `tests/unit/context/test_context_plugin_registry.py` | P0 | Planned | `-m gate` |

**Wave 1 exit:** `uv run pytest tests/unit/context/ -m gate -q` green.

---

### Wave 2 — DefaultNexusContextEngine unification (P0)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-3.1** | `DefaultNexusContextEngine` in `context_engine.py` implementing `ContextEngine` | P0 | Planned | Delegates to existing steps + compiler |
| **CE-3.2** | Adapter: `ContextCandidate` ↔ `ContextFragment` bridge | P0 | Planned | Round-trip tests |
| **CE-3.3** | `resolve_context_engine_from_environment()` in `context_wiring.py` | P0 | Planned | Preset `default` returns engine instance |
| **CE-3.4** | `CompileContextStep` accepts injectable `ContextEngine` / `ContextBudgetAllocator` | P0 | Planned | Closes GAP-CTX-03 |
| **CE-3.5** | `NexusLoop` + `nexus_factory` accept `context_engine` param | P1 | Planned | Back-compat: `context_manager` still works |
| **CE-3.6** | Document rename: `ContextBuilder` → `SessionRagContextBuilder` (alias deprecated one release) | P3 | Planned | Closes GAP-CTX-11 |
| **CE-3.7** | Graph path: `ContextManager.build_agent_context` calls `ContextEngine.assemble(scope=graph_node)` | P0 | Planned | Single code path; closes GAP-CTX-02 |
| **CE-3.8** | Integration test: turn + graph produce `CONTEXT_ASSEMBLED` with `engine_id=default` | P0 | Planned | `tests/integration/runtime/test_context_engine_paths.py` |

**Wave 2 exit:** Dual paths unified under `DefaultNexusContextEngine`; existing context unit tests green.

---

### Wave 3 — Step-aware assembly (P1)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-4.1** | Populate `ContextAssemblyRequest` in UAEP `BEFORE_CONTEXT_BUILD` with `step_index`, `step_kind` | P1 | Planned | Sourced from `AgentStepContext` |
| **CE-4.2** | ACP: map `StepOutcome` / contract `context_hints` → `required_sources` / `excluded_sources` | P1 | Planned | Agent contract optional field — no breaking change |
| **CE-4.3** | `objective` field from task message or active plan slice (`DecisionRecord` link) | P2 | Planned | REASONING plan cross-link only |
| **CE-4.4** | `DefaultContextRanker` boosts fragments matching `step_kind` (config table) | P1 | Planned | e.g. `tool_call` → boost TOOL_OUTPUT |
| **CE-4.5** | Event payload v2: `context_assembly.v2` with step fields (registry bump) | P1 | Planned | `payload_registry.py` + gate |
| **CE-4.6** | Unit tests step-aware ranking | P1 | Planned | `-m gate` |

**Wave 3 exit:** `CONTEXT_ASSEMBLED` payloads include `step_kind` on UAEP path.

---

### Wave 4 — Codebase preset and orchestrator (P1/P2)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-7.1** | Promote `workspace_index_spike.py` → `intergrax/context/providers/workspace_index.py` | P1 | Planned | AST chunking hook interface |
| **CE-7.2** | `WorkspaceContextProvider` implements `ContextSourceProvider` | P1 | Planned | Incremental Merkle root in metadata |
| **CE-7.3** | `CodebaseContextEngine` subclass — preset ranker (path proximity, import graph heuristic) | P1 | Planned | Reference in `applications/lab/` |
| **CE-7.4** | `ContextProfile.engine_preset="codebase"` wiring | P1 | Planned | Lab host opt-in |
| **CE-7.5** | Integration test: 1k file workspace → bounded context under budget | P1 | Planned | Acceptance `-m context_acceptance` |
| **CE-8.1** | `ContextOrchestrator` — bounded multi-hop collect (max_hops, latency_budget_ms) | P2 | Planned | Policy on hop count |
| **CE-8.2** | Wire orchestrator into `codebase` preset only | P2 | Planned | Default preset unchanged |
| **CE-8.3** | Explore delegation handoff: child preset `explore_child` auto-selected | P2 | Planned | NEXUS_EXECUTION_FLOW §27 link |

**Wave 4 exit:** Lab host demonstrates codebase preset with workspace provider.

---

### Wave 5 — Observability and quality hot path (P2)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-9.1** | Runtime events `CONTEXT_CANDIDATE_COLLECTED`, `CONTEXT_CANDIDATE_DROPPED`, `CONTEXT_VALIDATION_FAILED` | P2 | Planned | `canonical.py` payloads + registry |
| **CE-9.2** | OTel spans: `context.engine.assemble`, `context.provider.collect`, `context.budget.allocate` | P2 | Planned | `check_observability_gates.py` rows |
| **CE-9.3** | Structured logging `intergrax.context.engine` — no content at INFO | P2 | Planned | Log fixture test |
| **CE-9.4** | Metrics counters in `runtime/observability/context_counters.py` | P2 | Planned | Opt-in env + documented default |
| **CE-9.5** | Cost attribution hook when semantic compression calls LLM | P2 | Planned | V-COST event fields |
| **CE-10.1** | `DefaultContextRanker` integrates `evaluate_context_engineering()` thresholds | P2 | Planned | Closes GAP-CTX-05 |
| **CE-10.2** | Dedup by `content_hash` in collect merge phase | P2 | Planned | Suppression reasons in events |
| **CE-10.3** | Replace string-heuristic `classify_candidates` with provider-supplied metadata | P2 | Planned | Closes GAP-CTX-08 |
| **CE-10.4** | Extend `context_regression_benchmark.py` for engine presets | P2 | Planned | Baseline JSON per preset |
| **CE-10.5** | Acceptance `test_acceptance_context_compiler_long_session.py` updated for engine | P2 | Planned | Never-overflow invariant |

**Wave 5 exit:** OBS gates include CE spans; quality scoring on default ranker.

---

### Wave 6 — Tier-3 DX, scaffold, CI gates (P2)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-11.1** | `production_context_profile()` + `codebase_context_profile()` helpers | P2 | Planned | `applications/_shared/context_presets.py` |
| **CE-11.2** | Reference hosts: lab + poc_template wire `context_plugin_ids` | P2 | Planned | H-APP pattern |
| **CE-11.3** | `regulated_minimal` preset for legal_application | P2 | Planned | Document in host ARCHITECTURE |
| **CE-12.1** | `EXTENSION_AUTHOR_GUIDE.md` §4 Context plugin catalog | P2 | Planned | Parallel to Tool/Skill |
| **CE-12.2** | `AGENT_CREATION_GUIDE.md` Appendix L — link CE canon + custom engine example | P2 | Planned | |
| **CE-12.3** | Scaffold `new-application` emits `context_plugin` stub optional flag | P3 | Planned | |
| **CE-12.4** | `scripts/check_context_engine_wiring.py` — hosts with CE profile must resolve engine | P2 | Planned | CI gate |
| **CE-12.5** | `intergrax doctor` hint for missing context catalog bootstrap | P3 | Planned | |
| **CE-12.6** | Journal + FAUDIT layer 16 re-run evidence | P2 | Planned | L3+ score |

**Wave 6 exit:** Extension guide complete; CI gate for wiring.

---

## Master deliverables register (CE-EXT)

| ID | Wave | Status |
|----|------|--------|
| CE-1.1 | 1 | Planned |
| CE-1.2 | 1 | Planned |
| CE-1.3 | 1 | Planned |
| CE-1.4 | 1 | Planned |
| CE-1.5 | 1 | Planned |
| CE-1.6 | 1 | Planned |
| CE-2.1 | 1 | Planned |
| CE-2.2 | 1 | Planned |
| CE-2.3 | 1 | Planned |
| CE-2.4 | 1 | Planned |
| CE-2.5 | 1 | Planned |
| CE-3.1 | 2 | Planned |
| CE-3.2 | 2 | Planned |
| CE-3.3 | 2 | Planned |
| CE-3.4 | 2 | Planned |
| CE-3.5 | 2 | Planned |
| CE-3.6 | 2 | Planned |
| CE-3.7 | 2 | Planned |
| CE-3.8 | 2 | Planned |
| CE-4.1 | 3 | Planned |
| CE-4.2 | 3 | Planned |
| CE-4.3 | 3 | Planned |
| CE-4.4 | 3 | Planned |
| CE-4.5 | 3 | Planned |
| CE-4.6 | 3 | Planned |
| CE-7.1 | 4 | Planned |
| CE-7.2 | 4 | Planned |
| CE-7.3 | 4 | Planned |
| CE-7.4 | 4 | Planned |
| CE-7.5 | 4 | Planned |
| CE-8.1 | 4 | Planned |
| CE-8.2 | 4 | Planned |
| CE-8.3 | 4 | Planned |
| CE-9.1 | 5 | Planned |
| CE-9.2 | 5 | Planned |
| CE-9.3 | 5 | Planned |
| CE-9.4 | 5 | Planned |
| CE-9.5 | 5 | Planned |
| CE-10.1 | 5 | Planned |
| CE-10.2 | 5 | Planned |
| CE-10.3 | 5 | Planned |
| CE-10.4 | 5 | Planned |
| CE-10.5 | 5 | Planned |
| CE-11.1 | 6 | Planned |
| CE-11.2 | 6 | Planned |
| CE-11.3 | 6 | Planned |
| CE-12.1 | 6 | Planned |
| CE-12.2 | 6 | Planned |
| CE-12.3 | 6 | Planned |
| CE-12.4 | 6 | Planned |
| CE-12.5 | 6 | Planned |
| CE-12.6 | 6 | Planned |

**Total CE-EXT:** 47 tasks (all Planned until first implementation PR).

---

## Inherited closeout (do not re-implement)

These items are **Done** under MEMORY / CTX / R-Context — CE-EXT **consumes** them:

| Capability | Owner phase | Module |
|------------|-------------|--------|
| `ContextCompiler` + ladder | MEM-DEPTH-1 | `context_compiler.py` |
| `ContextManager` v2 + provenance | R-Context / CTX | `context_manager.py` |
| `context_runtime_bridge` | CTX-1 | `context_runtime_bridge.py` |
| `context_wiring` | CTX-2 | `context_wiring.py` |
| `ContextProfile` model | H-APP | `environment_profile.py` |
| `CONTEXT_ASSEMBLED` / `CONTEXT_TRIMMED` | R-Context | `context_skill_recording.py` |
| Hooks `BEFORE/AFTER_CONTEXT_BUILD` | UAEP | `uaep.py`, `hook_registry.py` |
| Quality scoring utilities | V-CE | `context_engineering.py` |
| Regression benchmark | V-CE | `context_regression_benchmark.py` |

---

## Verification commands

```bash
# Domain unit tests (after CE-1)
uv run pytest tests/unit/context/ tests/unit/runtime/nexus/context/ -m gate -q

# Application wiring
uv run pytest tests/unit/applications/test_context_wiring.py tests/unit/applications/test_context_runtime_bridge.py -m gate -q

# Acceptance never-overflow
uv run pytest tests/acceptance/test_acceptance_context_compiler_long_session.py -q

# Platform gates
uv run pytest -m gate -q
python scripts/check_docs_domain_pairs.py
uv run python scripts/check_observability_gates.py
```

---

## Explicitly out of scope (CE-EXT)

| Item | Owner |
|------|-------|
| L4 adaptive context learning loops | `ADAPTIVE_HARNESS_INTELLIGENCE` |
| Mem0 SaaS auto-ingest | MEMORY MEM-8 |
| Phase K business agents | PLATFORM_FOUNDATION §6.3 |
| RAG retrieval algorithm changes | `plan/RAG.md` |
| New memory store types | `plan/MEMORY.md` |

---

## Suggested PR order

```text
CE-DOC.* (Done)
→ CE-1.1 → CE-1.3 → CE-1.4 → CE-2.1 → CE-2.3 → CE-2.5
→ CE-3.1 → CE-3.7 → CE-3.4 → CE-3.8
→ CE-4.1 → CE-4.4 → CE-4.5
→ CE-7.1 → CE-7.2 → CE-7.3 → CE-7.5
→ CE-9.1 → CE-9.2 → CE-10.1
→ CE-11.1 → CE-12.1 → CE-12.4
```

---

*End of Context Engineering implementation plan.*
