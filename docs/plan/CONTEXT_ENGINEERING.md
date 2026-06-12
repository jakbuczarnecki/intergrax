# Context Engineering — Implementation Plan

**Architecture (1:1):** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**ADR:** [`ADR-CTX-001`](../adr/entries/2026-06-12/ADR-CTX-001.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Status summary (2026-06-12)

| Phase | Scope | Status |
|-------|-------|--------|
| **CTX** (control plane closeout) | `context_runtime_bridge`, `context_wiring`, Appendix L | **Done** (2026-06-02) |
| **R-Context** | Budget API, `CONTEXT_*` events (graph path) | **Done** |
| **MEM-DEPTH-1.\*** | `ContextCompiler`, `DegradationLadder`, preflight **modules** | **Done** — library + tests; **hot-path wiring = CE-3.9** (post ACP-CLOSE) |
| **CE-DOC** | Domain split + architecture + plan + FAUDIT refresh | **Done** (CE-DOC.7 closes 2026-06-12 audit) |
| **CE-EXT** | Plugin engine + hot-path compiler + step-aware + codebase preset | **Done** (S0–S12, 2026-06-12) |
| **CE-DOC.8** | Architecture ↔ implementation sync post CE-EXT | **Done** (2026-06-12) |
| **CE-ALIGN** | Post-audit implementation alignment (GAP-CTX-15..19) | **Done** (A0–A6, 2026-06-12) |
| **CE-PROV-WIRE** | Builtin stub providers → legacy collectors on `assemble()` path | **Planned** — see [Phase CE-PROV-WIRE](#phase-ce-prov-wire) |
| **CE-DOC.9** | FAUDIT 2026-06-12 deep audit — GAP-CTX-15..19 + CE-ALIGN sprint register | **Done** (2026-06-12) |
| **CE-DOC.10** | CE-ALIGN closeout audit + architecture sync | **Done** (2026-06-12) |
| **CE-DOC.11** | CE-PROV-WIRE phase + GAP-CTX-20 register; architecture §8.4 sync | **Done** (2026-06-12) |

**As-built maturity:** L3+ engine spine (FORMAT merge + graph/UAEP assemble) · **L3− plugin collect** until CE-PROV-WIRE closes GAP-CTX-20; see architecture §3.

**Delivery rule:** One **CE-\*** ID per PR → update master table + gap register → `pytest -m gate` + domain CI scripts green.

---

## Gap traceability matrix (GAP-CTX → CE)

| GAP-CTX | CE IDs | Wave | Status |
|---------|--------|------|--------|
| GAP-CTX-01 | CE-1, CE-2 | 1 | **Closed** |
| GAP-CTX-02 | CE-3 | 2 | **Closed** (hybrid UAEP/ACP) |
| GAP-CTX-03 | CE-3.4 | 2 | **Closed** |
| GAP-CTX-04 | CE-4 | 3 | **Closed** |
| GAP-CTX-05 | CE-10.1 | 5 | **Closed** |
| GAP-CTX-06 | CE-7 | 4 | **Closed** |
| GAP-CTX-07 | CE-8 | 4 | **Closed** |
| GAP-CTX-08 | CE-10.3 | 5 | **Open** (deferred) |
| GAP-CTX-09 | CE-9.2 | 5 | **Closed** |
| GAP-CTX-10 | CE-9.1 | 5 | **Closed** |
| GAP-CTX-11 | CE-3.6 | 2 | **Closed** |
| GAP-CTX-12 | AHI | — | **Deferred** |
| GAP-CTX-13 | CE-3.9, CE-3.10 | 2 | **Closed** |
| GAP-CTX-14 | CE-3.11 | 2 | **Closed** |
| GAP-CTX-15 | CE-FMT-1, CE-FMT-2 | CE-ALIGN | **Closed** |
| GAP-CTX-16 | CE-8.2b, CE-PROV-CTX | CE-ALIGN | **Closed** |
| GAP-CTX-17 | CE-ENG-REF | CE-ALIGN | **Closed** |
| GAP-CTX-18 | CE-PRESET-ENG | CE-ALIGN | **Closed** |
| GAP-CTX-19 | CE-REGISTRY-FMT | CE-ALIGN | **Closed** (formatter via catalog; allocator = ContextCompiler) |
| GAP-CTX-20 | CE-PROV-WIRE | CE-PROV-WIRE | **Open** (partial) — 8 builtin `collect()` stubs remain; live: workspace, session_semantic, task_message, graph_prior, session_history (B1) |

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
| CE-DOC.7 | FAUDIT layer 16 refresh — post-ACP as-built paths, GAP-CTX-13/14, module inventory, sprint register | **Done** (2026-06-12) |
| CE-DOC.8 | Architecture canon sync with CE-EXT S0–S12 implementation (§2–§3, §8.3, §16–§17) | **Done** (2026-06-12) |
| CE-DOC.9 | Deep audit register GAP-CTX-15..19 + CE-ALIGN sprint plan | **Done** (2026-06-12) |
| CE-DOC.10 | CE-ALIGN closeout — architecture §2/§16 sync post A1–A6 | **Done** (2026-06-12) |
| CE-DOC.11 | CE-PROV-WIRE phase register + GAP-CTX-20; architecture §8.4/§16 sync | **Done** (2026-06-12) |

---

## Phase CE-PROV-WIRE — Builtin provider legacy collector wiring

**Status:** **Planned** (2026-06-12)  
**Goal:** Close **GAP-CTX-20** — every §8.4 builtin stub `collect()` delegates to the existing Nexus/Tier-0 collector for its source domain so `DefaultNexusContextEngine.assemble()` satisfies §7.1 on graph + UAEP paths (not only workspace/session_semantic).  
**Prerequisites:** CE-ALIGN Done (CE-FMT-1, CE-PROV-CTX handles) · CE-2.3 catalog  
**Success gate:** All 11 stub providers emit `ContextFragment[]` when handles populated; gate test per provider family; architecture §8.4 `collect status` = **live** or **live (handle-gated)**.

**Distinction:** **CE-PROV-CTX** (Done) passes handles into `ContextProviderContext`; **CE-PROV-WIRE** implements `collect()` body for each builtin stub.

| ID | Deliverable | Priority | Status | Legacy source (as-built) |
|----|-------------|----------|--------|--------------------------|
| **CE-PROV-BRIDGE** | `intergrax/context/providers/legacy_bridge.py` — shared adapters + handle key contract doc | P0 | **Done** (2026-06-12) | `provider_handles.py` extension |
| **CE-PROV-01** | `builtin.task_message` — objective + user turn from `messages` / request | P0 | **Done** (2026-06-12) | `graph_assembly` / `ContextAssemblyRequest.objective` |
| **CE-PROV-02** | `builtin.system_instructions` — system prompt slices from `runtime_config` / prompt registry handles | P1 | Planned | `RuntimeConfig` + agent contract prompt assets |
| **CE-PROV-03** | `builtin.session_history` — chronological session turns | P0 | **Done** (2026-06-12) | `session_history_messages` handle / task metadata |
| **CE-PROV-04** | `builtin.longterm_memory` — LTM search hits | P1 | Planned | MEMORY LTM step / `memory_view` handles |
| **CE-PROV-05** | `builtin.rag` — retrieved chunks + citations | P0 | Planned | `ContextBuilder` / `rag_chunks` handle / catalog retrieve snapshot |
| **CE-PROV-06** | `builtin.websearch` — websearch result blocks | P2 | Planned | websearch step output handle |
| **CE-PROV-07** | `builtin.tool_output` — tool result blocks from step | P1 | Planned | `tool_context_helpers.py` / step tool blocks handle |
| **CE-PROV-08** | `builtin.graph_prior` — prior node outputs on graph path | P0 | **Done** (2026-06-12) | `prior_output_records` handle from `collect_dependency_records` |
| **CE-PROV-09** | `builtin.shared_context` — `SharedTaskContext` KV reads | P1 | Planned | `shared_task_context.py` via task metadata |
| **CE-PROV-10** | `builtin.attachments` — attachment summaries | P2 | Planned | `AttachmentRef` / ingestion summaries handle |
| **CE-PROV-11** | `builtin.policy_overlay` — policy fragment bundles | P2 | Planned | `prompt_policy_overlay.py` / policy handles |
| **CE-PROV-GATE** | `scripts/check_context_builtin_providers.py` — no empty stub for wired provider ids | P1 | Planned | CI gate |
| **CE-PROV-INT** | Integration tests: graph assemble with RAG + graph_prior fragments in LLM window | P1 | Planned | `tests/integration/runtime/test_context_provider_wiring.py` |

**CE-2.3 completion note:** catalog registration is **Done**; **live collect** for all §8.4 rows completes when CE-PROV-WIRE is **Done**.

**Deferred with CE-PROV-WIRE:** CE-10.3 (provider metadata replaces `classify_candidates` heuristics) — run after CE-PROV-05/08 validate fragment metadata shape.

---

## Phase CE-ALIGN — Architecture ↔ implementation alignment

**Status:** **Done** (2026-06-12)  
**Goal:** Close GAP-CTX-15..19 discovered in post-CE-EXT FAUDIT — unified `collect → format → budget` spine on production paths.  
**Prerequisites:** CE-EXT Done · CE-DOC.9 Done  
**Success gate:** `DefaultNexusContextEngine` merges provider fragments into LLM window; codebase orchestrator on graph path; custom `engine_ref` resolves; preset engines behave per §8.5; final FAUDIT layer 16 green.

| ID | Deliverable | Priority | Status |
|----|-------------|----------|--------|
| **CE-FMT-1** | `DefaultContextFormatter` — ranked `ContextFragment[]` → `ChatMessage[]` merge before `ContextCompiler` | P0 | **Done** |
| **CE-FMT-2** | `fragments_excluded` populated from ranker quality gate + dedup reasons | P1 | **Done** |
| **CE-8.2b** | Wire `ContextOrchestrator.assemble_with_hops` on graph `codebase` preset | P1 | **Done** |
| **CE-ENG-REF** | Resolve `ContextProfile.engine_ref` custom class via dotted import | P1 | **Done** |
| **CE-PRESET-ENG** | `RegulatedMinimalContextEngine` + `ExploreChildContextEngine` ranker/threshold behavior | P2 | **Done** |
| **CE-REGISTRY-FMT** | Engine uses registry `formatter` when plugin sets it (`BuiltinContextPlugin`) | P2 | **Done** |
| **CE-PROV-CTX** | Graph `provider_ctx` handles: `workspace_files`, session vector flags from task/env | P1 | **Done** |
| **CE-7.5b** | Integration test: 1k-file workspace → assemble under token budget with fragment in window | P1 | **Done** |
| **CE-UAEP-ASM** | UAEP session turn optional `ContextEngine.assemble()` when engine wired | P2 | **Done** |
| **CE-HOOKS-GRAPH** | `BEFORE_CONTEXT_BUILD` / `AFTER_CONTEXT_BUILD` on graph engine assemble | P2 | **Done** |

**Deferred (unchanged):** CE-9.5, CE-9.6, CE-10.3–10.5, CE-12.1–12.3 · GAP-CTX-08 · GAP-CTX-12.

---

## Phase CE-EXT — Context Engineering Plugin Engine

**Status:** **Done** (2026-06-12)  
**Goal:** Close **GAP-CTX-\*** rows — production-grade plugin engine integrated with Harness observability, policy, and Tier-3 presets.  
**Prerequisites:** CE-DOC Done · MEM-DEPTH-1 Done · CTX Done · OBS event spine Done  
**Success gate:** L3+ engine on FAUDIT layer 16 · `codebase` preset shipped · `check_context_engine_wiring.py` green  
**Deferred follow-up:** CE-9.5, CE-9.6, CE-10.3–10.5, CE-12.1–12.3 · GAP-CTX-08

### Execution order (waves)

```text
Wave 1 (P0): CE-1 contracts + CE-2 registry + CE-2.6 ContextProfile fields
Wave 2 (P0): CE-3 engine + CE-3.9 hot-path compiler wiring (critical)
Wave 3 (P1): CE-4 step-aware + CE-5.1 contract hints + CE-VEC-1 episodic recall
Wave 4 (P1): CE-7 workspace provider + CE-8 orchestrator
Wave 5 (P2): CE-9 observability + CE-10 quality in hot path
Wave 6 (P2): CE-11 Tier-3 presets + CE-12 DX / scaffold / gates
```

See [Sprints (CE-EXT delivery)](#sprints-ce-ext-delivery) for operator-facing sprint breakdown.

---

### Wave 1 — Contracts and plugin registry (P0)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-1.1** | Create `intergrax/context/contracts.py` — `ContextFragment`, `ContextFragmentSource`, `AssembledContext`, `BudgetAllocationResult` | P0 | **Done** | Unit tests for frozen dataclasses / enums |
| **CE-1.2** | `ContextAssemblyRequest` + `ContextProviderContext` (runtime handles via typed ctx object) | P0 | **Done** | Schema version field; no secrets in repr |
| **CE-1.3** | Protocols: `ContextSourceProvider`, `ContextRanker`, `ContextBudgetAllocator`, `ContextFormatter`, `ContextValidator`, `ContextEngine` | P0 | **Done** | `typing.Protocol` + docstrings; gate import boundary test |
| **CE-1.4** | `ContextPlugin` dataclass + `ContextPluginRegistry` | P0 | **Done** | Register/list/unregister providers |
| **CE-1.5** | Move shared scoring types from `context_engineering.py` → `intergrax/context/quality.py` (re-export shim) | P1 | **Done** | Re-export shim in `context_engineering.py` |
| **CE-1.6** | Architecture gate: `intergrax/context/` MUST NOT import `agents/` or `applications/` | P0 | **Done** | `scripts/check_context_tier0_import_boundary.py` |
| **CE-2.1** | `register_context_plugin()` + `intergrax.context` entry point group in `pyproject.toml` | P0 | **Done** | `pyproject.toml` EP + `register_context_plugin()` |
| **CE-2.2** | `bootstrap_context_catalog()` in `intergrax/context/bootstrap.py` | P0 | **Done** | `wire_application_environment` calls bootstrap |
| **CE-2.3** | Shipped `BuiltinContextPlugin` registering all §8.4 providers (catalog stubs + live workspace/session) | P0 | **Done** (collect live → **CE-PROV-WIRE**) | 13 builtin provider ids |
| **CE-2.4** | `ContextProfile.context_plugin_ids` + validation against registry | P1 | **Done** | `validate_context_plugin_ids` — lab fail / strict warn |
| **CE-2.5** | Unit tests `tests/unit/context/test_context_plugin_registry.py` | P0 | **Done** | catalog + wiring tests `-m gate` |
| **CE-2.6** | Extend `ContextProfile` with `engine_preset`, `engine_ref`, `context_plugin_ids` | P0 | **Done** | `environment_profile.py`; `context_runtime_bridge.py` metadata |

**Wave 1 exit:** `uv run pytest tests/unit/context/ -m gate -q` green.

---

### Wave 2 — DefaultNexusContextEngine unification (P0)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-3.1** | `DefaultNexusContextEngine` in `context_engine.py` implementing `ContextEngine` | P0 | **Done** | Delegates to existing steps + compiler |
| **CE-3.2** | Adapter: `ContextCandidate` ↔ `ContextFragment` bridge | P0 | **Done** | Round-trip tests |
| **CE-3.3** | `resolve_context_engine_from_environment()` in `context_wiring.py` | P0 | **Done** | Preset `default` returns engine instance |
| **CE-3.4** | Injectable `ContextEngine` / `ContextBudgetAllocator` on assembly path (replaces legacy `CompileContextStep`) | P0 | **Done** | Closes GAP-CTX-03 |
| **CE-3.5** | `NexusLoop` + `nexus_factory` accept `context_engine` param | P1 | **Done** | Back-compat: `context_manager` still works |
| **CE-3.6** | Document rename: `ContextBuilder` → `SessionRagContextBuilder` (alias deprecated one release) | P3 | **Done** | Closes GAP-CTX-11 |
| **CE-3.7** | Graph path: `ContextManager.build_agent_context` calls `ContextEngine.assemble(scope=graph_node)` | P0 | **Done** | Single code path; closes GAP-CTX-02 |
| **CE-3.8** | Integration test: ACP + graph produce `CONTEXT_ASSEMBLED` with `engine_id=default` | P0 | **Done** | `tests/integration/runtime/test_context_engine_paths.py` |
| **CE-3.9** | Wire `ContextCompiler.compile()` + degradation ladder to ACP `on_next_step` / UAEP before LLM | **P0 Critical** | **Done** | Closes GAP-CTX-13; acceptance never-overflow on prod path |
| **CE-3.10** | `ContextValidator` → `verify_context_preflight()` inside `DefaultNexusContextEngine.validate` | P0 | **Done** | Gate: preflight before every engine-assembled LLM call |
| **CE-3.11** | Unify UAEP `CONTEXT_BUILT` → `CONTEXT_ASSEMBLED` (+ trim events); deprecate or alias `CONTEXT_BUILT` | P1 | **Done** | Closes GAP-CTX-14; `payload_registry.py` + gate |

**Wave 2 exit:** `ContextCompiler` on production hot path; graph + ACP/UAEP unified under `DefaultNexusContextEngine`.

---

### Wave 3 — Step-aware assembly (P1)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-4.1** | Populate `ContextAssemblyRequest` in UAEP `BEFORE_CONTEXT_BUILD` with `step_index`, `step_kind` | P1 | **Done** | Sourced from `AgentStepContext` |
| **CE-4.2** | ACP: map `StepOutcome` / contract `context_hints` → `required_sources` / `excluded_sources` | P1 | **Done** | Agent contract optional field — no breaking change |
| **CE-4.3** | `objective` field from task message or active plan slice (`DecisionRecord` link) | P2 | **Done** | REASONING plan cross-link only |
| **CE-4.4** | `DefaultContextRanker` boosts fragments matching `step_kind` (config table) | P1 | **Done** | e.g. `tool_call` → boost TOOL_OUTPUT |
| **CE-4.5** | Event payload v2: `context_assembly.v2` with step fields (registry bump) | P1 | **Done** | `payload_registry.py` + gate |
| **CE-4.6** | Unit tests step-aware ranking | P1 | **Done** | `-m gate` |
| **CE-4.7** | `pre_context_policy_audit` inside `assemble()` before format/validate | P1 | **Done** | `pre_context_policy_audit.py` wired; hook parity unchanged |
| **CE-5.1** | Optional `AgentContract.context_hints` → `required_sources` / `excluded_sources` | P2 | **Done** | `AGENT_CONTRACTS` plan cross-link; CE-4.2 consumes |
| **CE-VEC-1** | `SessionSemanticRecallProvider` + `SESSION_HISTORY_SEMANTIC` in ranker/degradation | P1 | **Done** | Gated on `MemoryProfile.enable_session_vector_index` + provider handles |

**Wave 3 exit:** `CONTEXT_ASSEMBLED` v2 payloads include `step_kind` on ACP/UAEP paths; episodic recall fragment when MEM-VEC enabled.

---

### Wave 4 — Codebase preset and orchestrator (P1/P2)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-7.1** | Promote `workspace_index_spike.py` → `intergrax/context/providers/workspace_index.py` | P1 | **Done** | AST chunking hook interface |
| **CE-7.2** | `WorkspaceContextProvider` implements `ContextSourceProvider` | P1 | **Done** | Incremental Merkle root in metadata |
| **CE-7.3** | `CodebaseContextEngine` subclass — preset ranker (path proximity, import graph heuristic) | P1 | **Done** | `codebase_engine.py` |
| **CE-7.4** | `ContextProfile.engine_preset="codebase"` wiring | P1 | **Done** | `resolve_context_engine_from_environment` |
| **CE-7.5** | Integration test: 1k file workspace → bounded context under budget | P1 | **Done** | `test_ce_s6_s12_modules.py` gate |
| **CE-8.1** | `ContextOrchestrator` — bounded multi-hop collect (max_hops, latency_budget_ms) | P2 | **Done** | `context/orchestrator.py` |
| **CE-8.2** | Wire orchestrator into `codebase` preset only | P2 | **Done** | `resolve_context_orchestrator_from_environment` |
| **CE-8.3** | Explore delegation handoff: child preset `explore_child` auto-selected | P2 | **Done** | Graph `ContextManager` delegation swap |

**Wave 4 exit:** Lab host demonstrates codebase preset with workspace provider.

---

### Wave 5 — Observability and quality hot path (P2)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-9.1** | Runtime events `CONTEXT_CANDIDATE_COLLECTED`, `CONTEXT_CANDIDATE_DROPPED`, `CONTEXT_VALIDATION_FAILED` | P2 | **Done** | Enum + payload + engine bus emission on graph assemble |
| **CE-9.2** | OTel spans: `context.engine.assemble`, `context.provider.collect`, `context.budget.allocate` | P2 | **Done** | `check_context_otel_span_registry.py` |
| **CE-9.3** | Structured logging `intergrax.context.engine` — no content at INFO | P2 | **Done** | Engine assemble logs scope metadata only |
| **CE-9.4** | Metrics counters in `runtime/observability/context_counters.py` | P2 | **Done** | `INTERGRAX_CONTEXT_METRICS` opt-in |
| **CE-9.5** | Cost attribution hook when semantic compression calls LLM | P2 | Deferred | Awaiting semantic compression hot path |
| **CE-9.6** | OBS product dashboard — context assembly SLO panel | P3 | Deferred | Link when OBS dashboard slice ships |
| **CE-10.1** | `DefaultContextRanker` integrates `evaluate_context_engineering()` thresholds | P2 | **Done** | Closes GAP-CTX-05 |
| **CE-10.2** | Dedup by `content_hash` in collect merge phase | P2 | **Done** | `context/dedup.py` in engine assemble |
| **CE-10.3** | Replace string-heuristic `classify_candidates` with provider-supplied metadata | P2 | Deferred | Post engine-unification follow-up |
| **CE-10.4** | Extend `context_regression_benchmark.py` for engine presets | P2 | Deferred | Baseline JSON per preset |
| **CE-10.5** | Acceptance `test_acceptance_context_compiler_long_session.py` updated for engine | P2 | Deferred | Engine path covered by gate unit tests |

**Wave 5 exit:** OBS gates include CE spans; quality scoring on default ranker.

---

### Wave 6 — Tier-3 DX, scaffold, CI gates (P2)

| ID | Deliverable | Priority | Status | Acceptance |
|----|-------------|----------|--------|------------|
| **CE-11.1** | `production_context_profile()` + `codebase_context_profile()` helpers | P2 | **Done** | `applications/_shared/context_presets.py` |
| **CE-11.2** | Reference hosts: lab + poc_template wire `context_plugin_ids` | P2 | **Done** | Lab `production_context_profile` wiring |
| **CE-11.3** | `regulated_minimal` preset for legal_application | P2 | **Done** | `regulated_minimal_context_profile()` helper |
| **CE-11.4** | `explore_child_context_profile()` helper + delegation auto-wire doc | P2 | **Done** | Preset + graph delegation swap |
| **CE-12.1** | `EXTENSION_AUTHOR_GUIDE.md` §4 Context plugin catalog | P2 | Deferred | Use `intergrax/context/plugin.py` canon until guide slice |
| **CE-12.2** | `AGENT_CREATION_GUIDE.md` Appendix L — link CE canon + custom engine example | P2 | Deferred | Appendix L exists — expand in DX slice |
| **CE-12.3** | Scaffold `new-application` emits `context_plugin` stub optional flag | P3 | Deferred | Scaffold follow-up |
| **CE-12.4** | `scripts/check_context_engine_wiring.py` — hosts with CE profile must resolve engine | P2 | **Done** | CI gate + `intergrax doctor` |
| **CE-12.5** | `intergrax doctor` hint for missing context catalog bootstrap | P3 | **Done** | `check_context_engine_wiring` in doctor |
| **CE-12.6** | Journal + FAUDIT layer 16 re-run evidence | P2 | **Done** | Journal + audit guide refresh 2026-06-12 |

**Wave 6 exit:** Extension guide complete; CI gate for wiring.

---

## Master deliverables register (CE-EXT)

| ID | Wave | Status |
|----|------|--------|
| CE-1.1 | 1 | **Done** |
| CE-1.2 | 1 | **Done** |
| CE-1.3 | 1 | **Done** |
| CE-1.4 | 1 | **Done** |
| CE-1.5 | 1 | **Done** |
| CE-1.6 | 1 | **Done** |
| CE-2.1 | 1 | **Done** |
| CE-2.2 | 1 | **Done** |
| CE-2.3 | 1 | **Done** |
| CE-2.4 | 1 | **Done** |
| CE-2.5 | 1 | **Done** |
| CE-2.6 | 1 | **Done** |
| CE-3.1 | 2 | **Done** |
| CE-3.2 | 2 | **Done** |
| CE-3.3 | 2 | **Done** |
| CE-3.4 | 2 | **Done** |
| CE-3.5 | 2 | **Done** |
| CE-3.6 | 2 | **Done** |
| CE-3.7 | 2 | **Done** |
| CE-3.8 | 2 | **Done** |
| CE-3.9 | 2 | **Done** |
| CE-3.10 | 2 | **Done** |
| CE-3.11 | 2 | **Done** |
| CE-4.1 | 3 | **Done** |
| CE-4.2 | 3 | **Done** |
| CE-4.3 | 3 | **Done** |
| CE-4.4 | 3 | **Done** |
| CE-4.5 | 3 | **Done** |
| CE-4.6 | 3 | **Done** |
| CE-4.7 | 3 | **Done** |
| CE-5.1 | 3 | **Done** |
| CE-VEC-1 | 3 | **Done** |
| CE-7.1 | 4 | **Done** |
| CE-7.2 | 4 | **Done** |
| CE-7.3 | 4 | **Done** |
| CE-7.4 | 4 | **Done** |
| CE-7.5 | 4 | **Done** |
| CE-8.1 | 4 | **Done** |
| CE-8.2 | 4 | **Done** |
| CE-8.3 | 4 | **Done** |
| CE-9.1 | 5 | **Done** |
| CE-9.2 | 5 | **Done** |
| CE-9.3 | 5 | **Done** |
| CE-9.4 | 5 | **Done** |
| CE-9.5 | 5 | Deferred |
| CE-9.6 | 5 | Deferred |
| CE-10.1 | 5 | **Done** |
| CE-10.2 | 5 | **Done** |
| CE-10.3 | 5 | Deferred |
| CE-10.4 | 5 | Deferred |
| CE-10.5 | 5 | Deferred |
| CE-11.1 | 6 | **Done** |
| CE-11.2 | 6 | **Done** |
| CE-11.3 | 6 | **Done** |
| CE-11.4 | 6 | **Done** |
| CE-12.1 | 6 | Deferred |
| CE-12.2 | 6 | Deferred |
| CE-12.3 | 6 | Deferred |
| CE-12.4 | 6 | **Done** |
| CE-12.5 | 6 | **Done** |
| CE-12.6 | 6 | **Done** |

**Total CE-EXT:** 57 tasks — **S0–S12 complete** (8 items deferred: CE-9.5, CE-9.6, CE-10.3–10.5, CE-12.1–12.3).

---

## Inherited closeout (do not re-implement)

These items are **Done** under MEMORY / CTX / R-Context — CE-EXT **consumes** them:

| Capability | Owner phase | Module |
|------------|-------------|--------|
| `ContextCompiler` + ladder (modules) | MEM-DEPTH-1 | `context_compiler.py`, `degradation_ladder.py` — **re-wire hot path in CE-3.9** |
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

# Context Tier-0 import boundary (CE-1.6)
python scripts/check_context_tier0_import_boundary.py

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
→ CE-2.6 → CE-1.1 → CE-1.3 → CE-1.4 → CE-2.1 → CE-2.3 → CE-2.5
→ CE-3.1 → CE-3.9 → CE-3.10 → CE-3.7 → CE-3.4 → CE-3.11 → CE-3.8
→ CE-4.1 → CE-5.1 → CE-4.4 → CE-4.5 → CE-4.7
→ CE-VEC-1 (after MEM-VEC-2.1)
→ CE-7.1 → CE-7.2 → CE-7.3 → CE-7.5
→ CE-9.1 → CE-9.2 → CE-10.1
→ CE-11.1 → CE-11.4 → CE-12.1 → CE-12.4
```

---

## Sprints (CE-EXT delivery)

Operator-facing sprint plan. One sprint = one coherent PR batch (1–5 CE IDs). Adjust velocity to team capacity.

| Sprint | Goal | CE IDs | Exit criteria | Depends on |
|--------|------|--------|---------------|------------|
| **S0** | Documentation + audit alignment | CE-DOC.7 | Architecture §2–§17 reflect post-ACP as-built; GAP-CTX-13/14 registered | — |
| **S1** | Tier-0 contracts + profile fields | CE-2.6, CE-1.1–CE-1.4, CE-1.6 | `intergrax/context/contracts.py` exists; `ContextProfile` preset fields; gate import boundary | **Done** (2026-06-12) |
| **S2** | Plugin catalog bootstrap | CE-2.1–CE-2.5, CE-1.5 | `register_context_plugin()`, `BuiltinContextPlugin` ≥10 providers; `pytest tests/unit/context/` green | **Done** (2026-06-12) |
| **S3** | Engine skeleton + hot-path compiler | CE-3.1, CE-3.2, CE-3.9, CE-3.10 | `DefaultNexusContextEngine`; `ContextCompiler` on ACP/UAEP before LLM; acceptance never-overflow on prod path | **Done** (2026-06-12) |
| **S4** | Path unification + events | CE-3.3, CE-3.4, CE-3.7, CE-3.11, CE-3.8 | Graph + ACP share `assemble()`; unified `CONTEXT_ASSEMBLED`; integration test green | **Done** (2026-06-12) |
| **S5** | Step-aware assembly | CE-4.1–CE-4.6, CE-4.7, CE-5.1 | `step_kind` in payload v2; policy audit in `assemble()`; optional contract `context_hints` | **Done** (2026-06-12) |
| **S6** | Episodic vector recall | CE-VEC-1 | `SESSION_HISTORY_SEMANTIC` fragments when `MemoryProfile.enable_session_vector_index` | **Done** (2026-06-12) |
| **S7** | Codebase preset | CE-7.1–CE-7.5 | Provider + index modules; **e2e assemble under budget = CE-7.5b (CE-ALIGN)** | **Partial** (2026-06-12) |
| **S8** | Multi-hop orchestrator | CE-8.1–CE-8.3, CE-11.4 | `ContextOrchestrator` class + resolve helper; **hot-path wiring = CE-8.2b (CE-ALIGN)** | **Partial** (2026-06-12) |
| **S9** | Observability spine | CE-9.1–CE-9.6 | `CONTEXT_CANDIDATE_*` events; OTel spans in OBS gates; assembly SLO panel doc link | **Done** (2026-06-12; CE-9.5/9.6 deferred) |
| **S10** | Quality in hot path | CE-10.1–CE-10.5 | `DefaultContextRanker` + dedup; regression baselines per preset; no string-heuristic `classify_candidates` | **Done** (2026-06-12; CE-10.3–10.5 deferred) |
| **S11** | Tier-3 presets + DX | CE-11.1–CE-11.4, CE-12.1–CE-12.6, CE-3.5–CE-3.6 | Reference hosts wired; extension guide §Context; `check_context_engine_wiring.py`; FAUDIT L3+ evidence | **Done** (2026-06-12; CE-12.1–12.3 deferred) |
| **S12** | Closeout | CE-12.6 + journal | Layer 16 re-audit; all GAP-CTX-\* Closed or deferred (AHI) | **Done** (2026-06-12) |

**Critical path:** S1 → S2 → **S3** (hot-path compiler) → S4 → S5. S6 parallel after MEM-VEC. S7–S8 optional product slice.

**Minimum viable CE (MVP):** S0–S4 closes GAP-CTX-01, 02, 03, 13, 14 — L3 engine on default preset.

---

## Sprints (CE-ALIGN delivery)

Post-audit alignment sprints. One sprint = one commit.

| Sprint | Goal | CE IDs | Exit criteria |
|--------|------|--------|---------------|
| **A0** | Audit documentation | CE-DOC.9 | GAP-CTX-15..19 in architecture §16 + this register | **Done** |
| **A1** | Fragment format merge | CE-FMT-1, CE-FMT-2 | Ranked fragments injected into messages before compile; excluded tuple populated | **Done** |
| **A2** | Orchestrator hot path | CE-8.2b, CE-PROV-CTX | Codebase graph assemble uses orchestrator; workspace handles on provider_ctx | **Done** |
| **A3** | Custom engine + presets | CE-ENG-REF, CE-PRESET-ENG | `engine_ref` resolves; regulated_minimal / explore_child engines | **Done** |
| **A4** | Registry + e2e test | CE-REGISTRY-FMT, CE-7.5b | Registry formatter used; 1k workspace assemble gate test | **Done** |
| **A5** | UAEP + graph hooks | CE-UAEP-ASM, CE-HOOKS-GRAPH | Optional UAEP assemble; hooks on graph engine path | **Done** |
| **A6** | Closeout audit | CE-DOC.10 | Re-audit; GAP-CTX-15..19 Closed; journal | **Done** |

---

## Sprints (CE-PROV-WIRE delivery)

One sprint = one coherent provider family or gate. One commit per sprint.

| Sprint | Goal | CE IDs | Exit criteria |
|--------|------|--------|---------------|
| **B0** | Plan + architecture register | CE-DOC.11 | GAP-CTX-20 + phase CE-PROV-WIRE in plan/architecture §16 | **Done** |
| **B1** | Legacy bridge + graph core | CE-PROV-BRIDGE, CE-PROV-01, CE-PROV-08, CE-PROV-03 | Graph path: task_message, graph_prior, session_history fragments in assembled window | **Done** |
| **B2** | Memory + RAG | CE-PROV-04, CE-PROV-05, CE-PROV-06 | RAG/LTM/websearch fragments when handles set; gate unit tests |
| **B3** | Shared + tools + policy | CE-PROV-09, CE-PROV-07, CE-PROV-02, CE-PROV-11 | shared_context + tool_output + system/policy overlays |
| **B4** | Attachments + integration | CE-PROV-10, CE-PROV-INT, CE-PROV-GATE | Attachment summaries; integration test; `check_context_builtin_providers.py` green |

**Critical path:** B1 → B2 (unblocks CE-10.3 metadata follow-up).

---

*End of Context Engineering implementation plan.*
