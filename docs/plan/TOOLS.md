# Tools — Implementation Plan

**Architecture (1:1):** [`architecture/TOOLS.md`](../architecture/TOOLS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**Queue status (2026-06-12):** Phase **TOOL-ENG** **closed** (36/36) · [§Layer completion final audit](#layer-completion-final-audit-2026-06-12). Catalog expansion (Phase O / T-EXPAND) **closed**. Default harness queue → **gate maintenance** in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

**Layer completion mode (2026-06-12):** [§Layer completion audit](#layer-completion-audit-2026-06-12) · [§Layer completion sprints](#layer-completion-sprints-2026-06-12) · [§Final audit](#layer-completion-final-audit-2026-06-12)

---

## Layer completion audit (2026-06-12)

**Scope:** Tier-0 `intergrax/tools/` catalog + Tier-1 tool engine (`intergrax/runtime/nexus/tools/`, `intergrax/runtime/tools/`). Direct dependencies: INTEGRATIONS (compose only), SKILLS (selection), UNIFIED_EXECUTION_RUNTIME §42.12 (`ToolRuntime`), NEXUS_EXECUTION_FLOW §15–17.

**Maturity (audit map §11):** **L3** catalog · **L3** engine orchestration (2a) · **L3** atomic invoke (2b) · **L3** selection at scale · **L3** governance (HIGH+ verify, required-mode).

### Tier-0 — Tool Library (catalog)

| Area | Status | Evidence |
|------|--------|----------|
| `ToolContract` / `ToolRegistry` / `ToolPlugin` | **Done** | 190 tools · 48 bundles · `shipped_plugins.py` |
| MCP + OpenAI export | **Done** | `check_tool_mcp_schema_export.py` (via `uv run`) |
| Oversized-tool lint | **Done** | `check_oversized_tool_lint.py` |
| Legacy `ToolBase` | **Deprecated** | Phase O.7 Done |
| Injection defense wiring | **Done** | `check_tool_injection_defense.py` |
| Agent registry bypass | **Done** | `check_agent_registry_bypass.py` |

**Tier-0 verdict:** Production-ready catalog. No architectural gaps blocking hosts.

### Tier-1 — Tool engine

| Area | Status | Gap ID | Notes |
|------|--------|--------|-------|
| Planner bootstrap | **Done** | TOOL-ENG-0 | `planner_bootstrap.py` |
| Catalog dispatch + gateway | **Done** | TOOL-ENG-1/2 | ADR-TOOL-001 |
| Scope policy wiring | **Done** | TOOL-ENG-3 | `RuntimeContext.build()` |
| Plan constraints → planner | **Done** | TOOL-ENG-4 | `allowed_tool_ids` filter |
| `ToolSelectionStrategy` | **Done** | TOOL-ENG-5 | `strategy_for_mode()` · `test_tool_selection_strategy.py` |
| Bounded tool loop | **Done** | TOOL-ENG-6/18 | ADR-TOOL-002 · delegates to `BoundedReactPattern` |
| `tools_context_scope` | **Done** | TOOL-ENG-11 | `tool_planner_input.py` |
| `ToolInvocationPattern` plugin | **Done** | TOOL-ENG-16 | S1 — ADR-TOOL-003 |
| Shipped patterns (single / ReAct) | **Done** | TOOL-ENG-17–18,21–23 | S2–S3 |
| Parallel batch + aggregate | **Done** | TOOL-ENG-9,29 | S4 |
| Chain pattern | **Done** | TOOL-ENG-20 | S7 |
| Semantic selection | **Done** | TOOL-ENG-13 | S5 |
| Hierarchical selection | **Done** | TOOL-ENG-14 | S6 — v1 deterministic; LLM category pass deferred |
| Semantic parallel composite | **Done** | TOOL-ENG-25 | S5 |
| Selection strategy plugins | **Done** | TOOL-ENG-26/31 | S6 — instance override + EP loader |
| Selection telemetry | **Done** | TOOL-ENG-32 | S6 — `ToolSelectionDiagV1` |
| Post-tool verify HIGH+ | **Done** | TOOL-ENG-7 | Trace + enforce block; approval via `high_risk_tool_approvals` |
| `tools_mode=required` hard fail | **Done** | TOOL-ENG-8 | `ToolsRequiredError` |
| `tool_choice` wiring | **Done** | TOOL-ENG-12 | `tool_planning_policy.py` |
| Invoker unit tests | **Done** | TOOL-ENG-TEST.1 | S0 — `FakeRegistry` + `test_tool_selection_strategy.py` |

### Documentation alignment

| Item | Status |
|------|--------|
| TOOL-ENG-DOC.1–7 | **Done** |
| Engine gap register ↔ plan register | **Aligned** |
| Audit prompt `known_gaps` | **Aligned** — 2026-06-12 final audit |
| CI gates `check_tool_invocation_patterns` / `check_tool_engine_ahi_hook` | **Done** |

### Architecture violations

None detected in Tier-0 catalog handlers (no vendor SDK bypass). Tier-1 invoke path routes through `RuntimeToolInvoker`. Graph vs tool-pattern boundary documented (DOC.6).

---

## Layer completion final audit (2026-06-12)

**Krok 5 — audyt końcowy** po S0–S8. Weryfikacja: dokumentacja ↔ implementacja ↔ plan ↔ CI.

### Wynik

| Obszar | Werdykt |
|--------|---------|
| TOOL-ENG register (36/36) | **Closed** |
| Tier-0 catalog (190 tools) | **L3 Production** |
| Tier-1 selection L6 | **L3** — standard, keyword, semantic, hierarchical, plugins |
| Tier-1 orchestration 2a | **L3** — 5 shipped patterns + chain + EP registry |
| Tier-1 atomic invoke 2b | **L3** — dispatch, gateway, scope, idempotency |
| Governance | **L3** — required-mode, HIGH+ block, tool_choice |
| Dokumentacja architecture/plan | **Aligned** (post-final-audit sync) |
| Testy + CI | **Green** — see verification block below |

### Świadomie odroczone (nie blokują zamknięcia warstwy)

| Item | Uzasadnienie |
|------|--------------|
| Hierarchical LLM category pass | ADR-TOOL-005 v1 — deterministic rank wystarcza na L3 |
| L1 critic (`eval.judge`) per tool output | Osobny zakres CVL — opcjonalny via `CriticProfile` |
| Pusta grupa EP w `pyproject.toml` | Loadery gotowe; host pakiety rejestrują strategie/wzorce |

### Weryfikacja (2026-06-12)

```bash
uv run pytest tests/unit/runtime/nexus/tools/ -q                    # 58 passed
uv run python scripts/check_tool_invocation_patterns.py            # OK
uv run python scripts/check_tool_engine_ahi_hook.py                # OK
python scripts/check_legacy_tool_plan_booleans.py                  # OK
uv run python scripts/check_tool_mcp_schema_export.py              # OK
python scripts/check_tool_injection_defense.py                     # OK
python scripts/check_agent_registry_bypass.py                      # OK
```

**Warstwa Tools:** **ukończona** wg Layer Completion Mode.

---

## Layer completion sprints (2026-06-12)

Sprints close **Phase TOOL-ENG** remaining rows. One sprint ≈ one PR; update plan + architecture gap register after each.

| Sprint | IDs | Goal | Done when | Primary files |
|--------|-----|------|-----------|---------------|
| **S0** | TOOL-ENG-TEST.1 | Test hygiene | Invoker + selection unit tests green | `test_runtime_tool_invoker_*.py`, `test_tool_selection_strategy.py` |
| **S1** | TOOL-ENG-16, ADR-TOOL-003 | `ToolInvocationPattern` protocol | Conformance test + ADR Accepted | `tool_invocation_pattern.py`, `docs/adr/entries/2026-06-12/ADR-TOOL-003.md` |
| **S2** | TOOL-ENG-17,18,21,22 | Shipped single + ReAct patterns; config factory; loop delegation | `test_tool_loop_integration.py` green; factory unit test | `patterns/single_pass.py`, `patterns/bounded_react.py`, `tool_loop.py`, `config.py`, `config_types.py` |
| **S3** | TOOL-ENG-23 | Host profile bridge | `test_catalog_runtime_bridge.py` pattern field | `environment_profile.py`, `catalog_runtime_bridge.py` |
| **S4** | TOOL-ENG-29,9 | Parallel batch + aggregate | Wall-time test < serial sum | `patterns/parallel_batch.py`, `tool_invocation_aggregate.py` |
| **S5** | TOOL-ENG-13,25 | Semantic index + composite batch | Rank gate + integration test | `tool_catalog_embedder.py`, `patterns/parallel_semantic_batch.py` |
| **S6** | TOOL-ENG-14,15,31,26,32 | Hierarchical selection + plugins + telemetry | ADR-TOOL-004 · custom EP test | `hierarchical_tool_selector.py`, `tool_selection.py` |
| **S7** | TOOL-ENG-20,24,27,28,30 | Chain pattern + entry points + CI + lab DX | `check_tool_invocation_patterns.py` green | `patterns/deterministic_chain.py`, `applications/lab_application/` |
| **S8** | TOOL-ENG-7,8,12,10 | Governance closeout | Required-mode fail + HIGH verify | `tools_step.py`, `tool_verify_hooks.py` |

**Current execution:** S0–S8 **Done** (2026-06-12) · Phase TOOL-ENG **closed** (36/36 deliverables).

### S5 implementation spec (2026-06-12)

**Scope:** TOOL-ENG-13 + TOOL-ENG-25 — semantic tool index + composite parallel batch.

| Deliverable | Contract |
|-------------|----------|
| `ToolCatalogEmbedder` | `index_for_registry(registry)` → `ToolCatalogIndex`; embed via `BaseEmbeddingManager` |
| `ToolCatalogIndex.search_query` | Cosine top-k over registry subset; returns `(tool_id, score)` |
| `ToolSelectionMode.SEMANTIC` | `SemanticToolIndexSelectionStrategy`; requires `embedding_manager` on context |
| `ParallelSemanticBatchPattern` | Semantic top-k → auto `PlannedToolCall` → parallel invoke → aggregate |
| `pattern_for_mode(PARALLEL_SEMANTIC_BATCH)` | Returns `ParallelSemanticBatchPattern` |

**ADR:** [ADR-TOOL-004](../adr/entries/2026-06-12/ADR-TOOL-004.md)

**Acceptance:** `test_tool_catalog_embedder.py` rank gate · `test_parallel_semantic_batch_pattern.py`

### S6 implementation spec (2026-06-12)

**Scope:** TOOL-ENG-14,15,26,31,32 — hierarchical selection + plugin surfaces + telemetry.

| Deliverable | Contract |
|-------------|----------|
| `HierarchicalToolSelectionStrategy` | Category rank → tool rank; `ToolSelectionMode.HIERARCHICAL` |
| `tool_selection_max_hierarchy_passes` | Bounds category branches contributing tools |
| `tool_selection_strategy` / `tool_selection_strategy_id` | Instance override (A) and entry-point load (B) |
| `ToolSelectionDiagV1` | `ops:tool_selection` trace on planner allow-list resolve |

**ADR:** [ADR-TOOL-005](../adr/entries/2026-06-12/ADR-TOOL-005.md)

**Acceptance:** `test_hierarchical_tool_selector.py` · `test_tool_selection_registry.py` · `test_tool_selection_telemetry.py`

### S7 implementation spec (2026-06-12)

**Scope:** TOOL-ENG-20,24,27,28,30 — deterministic chain + invocation plugins + telemetry + CI + lab DX.

| Deliverable | Contract |
|-------------|----------|
| `ToolChainSpec` / `FieldRef` | Ordered steps with `input_mappings` |
| `DeterministicChainPattern` | Sequential invoke; no LLM between steps |
| `tool_invocation_registry.py` | `intergrax.tool_invocation_patterns` entry-point loader |
| `ToolsSummaryDiagV1` | `pattern_id`, `stop_reason`, `ops:tool_invocation_pattern` |
| `check_tool_invocation_patterns.py` | All `ToolInvocationMode` values ship via factory |
| `LAB_TOOL_INVOCATION_MODE` | Lab host env bridge to `ApplicationEnvironmentProfile` |

**ADR:** no ADR needed — extends ADR-TOOL-003 plugin model.

**Acceptance:** `test_deterministic_chain_pattern.py` · `test_tool_invocation_registry.py` · `test_lab_tool_invocation_mode.py`

### S8 implementation spec (2026-06-12)

**Scope:** TOOL-ENG-7,8,12,10 — governance and adaptive tool engine closeout.

| Deliverable | Contract |
|-------------|----------|
| `run_post_tool_verify` | Trace + `ToolVerificationRequiredError` when `enforce_high_risk_tool_verify` |
| `high_risk_tool_approvals` | `RuntimeState` bypass set for approved `tool_id`s |
| `ToolsRequiredError` | `tools_mode=required` with zero traces |
| `tool_choice_for_mode` | Maps `tools_mode` → planner `tool_choice` |
| `ToolEngineHook` + `recommend_tool_modes` | AHI routing hook picks L6/L2a per run |

**ADR:** no ADR needed — extends existing governance and AHI routing patterns.

**Acceptance:** `test_tool_verify_enforcement.py` · `test_adaptive_tool_mode_resolver.py` · `check_tool_engine_ahi_hook.py`

### S4 implementation spec (2026-06-12)

**Scope:** TOOL-ENG-29 + TOOL-ENG-9 — parallel read-only batch invoke + canonical aggregate.

| Deliverable | Contract |
|-------------|----------|
| `ToolInvocationAggregate` | `from_traces(traces) -> combined_context: str`, `success_count`, `failure_count`; stable merge order = planner call order |
| `execute_planned_tool_calls` | New `max_parallel_read_only: int` (default `1` = serial); partition by `ToolContract.side_effects` |
| `ParallelBatchPattern` | `pattern_id=parallel_batch`; planner single-pass → parallel read-only invoke → aggregate on `ToolInvocationResult` |
| `RuntimeConfig.max_parallel_tool_calls` | Default **8**; bridged from `ApplicationEnvironmentProfile` |
| `pattern_for_mode(PARALLEL_BATCH)` | Returns `ParallelBatchPattern` (no longer `NotImplementedError`) |

**Acceptance:** `test_tool_invocation_aggregate.py` · `test_parallel_batch_pattern.py` (3 read-only tools wall time < serial sum; mutating stays serial).

**ADR:** no ADR needed — extends ADR-TOOL-003 plugin model; no new Tier boundary.

---

## Phase TOOL-ENG — Tool engine hardening (2026-06-10 audit · extended 2026-06-12)

**Status:** **Closed** (2026-06-12) — **36/36** TOOL-ENG deliverables Done  
**Architecture canon:** [`architecture/TOOLS.md`](../architecture/TOOLS.md) — [Tool engine production posture](../architecture/TOOLS.md#tool-engine-production-posture-2026-06-10), [Invocation patterns](../architecture/TOOLS.md#tool-invocation-patterns-production-orchestration), [Engine gap register](../architecture/TOOLS.md#engine-gap-register-canon)  
**Audit basis:** Full-stack tool layer audit 2026-06-10 (Tier-0 catalog + Tier-1 selection/invoke/verify) · **Invocation-pattern audit 2026-06-12** (single / parallel / chain / graph + plugin contract)  
**Priority ladder:** **Band 2ba** — supersedes ad-hoc tool engine fixes until TOOL-ENG P0 closed  
**ADR:** [ADR-TOOL-001](../adr/entries/2026-06-10/ADR-TOOL-001.md) (TOOL-ENG-1/2) · [ADR-TOOL-002](../adr/entries/2026-06-11/ADR-TOOL-002.md) (TOOL-ENG-6) · [ADR-TOOL-003](../adr/entries/2026-06-12/ADR-TOOL-003.md) (TOOL-ENG-16) · [ADR-TOOL-004](../adr/entries/2026-06-12/ADR-TOOL-004.md) · [ADR-TOOL-005](../adr/entries/2026-06-12/ADR-TOOL-005.md)

**Problem statement (2026-06-10, partially closed):** Tier-0 catalog is production-grade (190 tools, contracts, MCP). Tier-1 gaps on bootstrap, dispatch, gateway, selection — **closed** TOOL-ENG-0–6,11.

**Problem statement (2026-06-12 — invocation patterns) — closed:** Tier-1 orchestration of multi-call plans was hardcoded in `run_bounded_tool_loop` → sequential `execute_planned_tool_calls`. **Resolved** by TOOL-ENG-16–30:

- `ToolInvocationPattern` plugin protocol (ADR-TOOL-003)
- Shipped patterns: single-pass, parallel batch, bounded ReAct, deterministic chain, parallel semantic batch
- Parallel read-only invoke (TOOL-ENG-9), semantic parallel composite (TOOL-ENG-25), `ToolChainSpec` chains (TOOL-ENG-20)
- Host wiring + entry-point registry for custom patterns (TOOL-ENG-23/24)
- Agent `ExecutionGraph` vs tool orchestration boundary documented (TOOL-ENG-DOC.6)

### TOOL-ENG — Master register

#### Selection & planning (L6–L6b)

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TOOL-ENG-0 | Bootstrap | **Wire `CatalogToolPlanner`** in `RuntimeContext.build` / catalog bridge | **Done** | **P0** | `planner_bootstrap.py`, `runtime_context.py`, `catalog_runtime_bridge.py` | `test_tool_engine_bootstrap.py` |
| TOOL-ENG-4 | Selection | **Plan constraints → planner** — `EnginePlan.tool_ids` allow-list | **Done** | P1 | `tools_step.py`, `tool_planning_service.py` | `test_tool_planning_constraints.py` |
| TOOL-ENG-5 | Selection | **`ToolSelectionStrategy` protocol** — static \| skill_pack \| retrieval_top_k \| full_catalog | **Done** | P1 | `tool_selection.py`, `tools_step.py` | `test_tool_selection_strategy.py` |
| TOOL-ENG-11 | Config | **`tools_context_scope`** — planner input per scope enum | **Done** | P1 | `tool_planner_input.py`, `tools_step.py` | scope enum tests |
| TOOL-ENG-13 | Selection | **Semantic tool index** — `ToolCatalogEmbedder`, `SemanticToolIndexSelectionStrategy`, reindex on catalog change | **Done** | P1 | `tool_catalog_embedder.py`, `tool_selection.py`, RAG `embedding_manager` | Gate: 190-tool query ranks correct `tool_id`; trace scores · **ADR required** |
| TOOL-ENG-14 | Selection | **Hierarchical tool selection** — category-tree multi-pass (v1 deterministic) | **Done** | P2 | `hierarchical_tool_selector.py`, `tool_selection.py` | `test_hierarchical_tool_selector.py` · **ADR-TOOL-005** |
| TOOL-ENG-15 | Selection | **`retrieval_top_k` naming** — keyword overlap clarity; optional `keyword_top_k` alias | **Done** | P2 | `config_types.py`, `catalog_runtime_bridge.py` | Alias accepted in bridge; enum alias test |
| TOOL-ENG-26 | Selection | **`ToolSelectionStrategy` entry-point plugins** — `intergrax.tool_selection_strategies` registry beyond `strategy_for_mode()` | **Done** | P2 | `tool_selection_registry.py` | `test_tool_selection_registry.py` · **ADR-TOOL-005** |
| TOOL-ENG-31 | Selection | **`RuntimeConfig.tool_selection_strategy` instance override** — inject custom strategy at bootstrap; overrides `tool_selection_mode` enum | **Done** | P1 | `config.py`, `plan_context_invocation.py` | Instance override test in registry suite |
| TOOL-ENG-32 | Selection | **Selection trace telemetry** — `strategy_id`, candidate `tool_id`s, scores in diag/trace | **Done** | P2 | `plan_context_invocation.py`, `tracing/tools/tool_selection.py` | `ops:tool_selection` on selection step |
| TOOL-ENG-DOC.7 | Docs | **Selection plugin model canon** — L6 protocol, shipped strategies, custom surfaces A/B/C | **Done** | P1 | `architecture/TOOLS.md` §selection plugin | §[Tool selection plugin model](../architecture/TOOLS.md#tool-selection-plugin-model-l6-extensibility) |

#### Dispatch & atomic invoke (2b)

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TOOL-ENG-1 | Dispatch | **Per-`tool_id` catalog dispatch** in `ToolRuntime.invoke` | **Done** | **P0** | `catalog_dispatch.py`, `tool_runtime.py` | `test_tool_runtime_catalog_dispatch.py`; ADR-TOOL-001 |
| TOOL-ENG-2 | Gateway | **Full-catalog `ToolRequest`** → `RuntimeToolInvoker` | **Done** | **P0** | `tool_gateway.py`, `catalog_dispatch.py` | `test_tool_gateway.py`; ADR-TOOL-001 |
| TOOL-ENG-3 | Policy | **`tool_scope_policy` → `RuntimeToolInvoker`** at `RuntimeContext.build()` | **Done** | **P0** | `runtime_context.py` | scope deny on invoke path |

#### Invocation orchestration (2a) — 2026-06-12 audit

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TOOL-ENG-DOC.5 | Docs | **Invocation patterns canon** — single / parallel / ReAct / chain / graph boundary + target `ToolInvocationPattern` contract | **Done** | P1 | `architecture/TOOLS.md`, FLOW §15.1, Appendix J, audit prompt | Architecture §[Invocation patterns](../architecture/TOOLS.md#tool-invocation-patterns-production-orchestration) |
| TOOL-ENG-16 | Orchestration | **`ToolInvocationPattern` Protocol** + `ToolInvocationContext` + `ToolInvocationResult` | **Done** | **P0** | `tool_invocation_pattern.py` (new), `contracts/` | `test_tool_invocation_pattern.py`; ADR-TOOL-003 |
| TOOL-ENG-17 | Orchestration | **`SinglePassPattern`** — extract `max_iterations==1` path from `tool_loop_step` | **Done** | P1 | `patterns/single_pass.py`, `tool_loop_step.py` | Behaviour parity with pre-refactor single-pass; unit test |
| TOOL-ENG-18 | Orchestration | **`BoundedReactPattern`** — refactor multi-iter `run_bounded_tool_loop` into pattern class | **Done** | P1 | `patterns/bounded_react.py`, `tool_loop_step.py` | `test_tool_loop_integration.py` unchanged green |
| TOOL-ENG-9 | Orchestration | **`ParallelBatchPattern`** — concurrent invoke for `side_effects=False` in one batch; `max_parallel_tool_calls` cap | **Done** | P1 | `patterns/parallel_batch.py`, `tool_loop.py` | `test_parallel_batch_pattern.py` |
| TOOL-ENG-20 | Orchestration | **`DeterministicChainPattern`** + `ToolChainSpec` / `ChainStep` / `FieldRef` output→input mapping | **Done** | P2 | `patterns/deterministic_chain.py`, `tool_chain_spec.py` | `test_deterministic_chain_pattern.py` |
| TOOL-ENG-25 | Orchestration | **`ParallelSemanticBatchPattern`** — semantic top-k selection + parallel invoke + aggregate (composite) | **Done** | P1 | `patterns/parallel_semantic_batch.py`, `tool_selection.py` | Depends TOOL-ENG-13+9+29; integration test with fixture index |
| TOOL-ENG-29 | Orchestration | **`ToolInvocationAggregate`** — canonical batch result merge before LLM context inject | **Done** | P1 | `tool_invocation_aggregate.py`, `tool_loop.py` | `test_tool_invocation_aggregate.py` |
| TOOL-ENG-21 | Config | **`RuntimeConfig.tool_invocation_pattern`** + `ToolInvocationMode` enum + `pattern_for_mode()` factory | **Done** | P1 | `config.py`, `config_types.py`, `tool_invocation_pattern.py` | Default `single_pass`; factory returns correct pattern class |
| TOOL-ENG-22 | Wiring | **`run_bounded_tool_loop` / `ctx.invoke_tool` delegates to `ToolInvocationPattern`** — remove direct `run_bounded_tool_loop` call | **Done** | P1 | `tools_step.py`, `runtime_context.py` | Inject pattern at `RuntimeContext.build()`; existing tests green |
| TOOL-ENG-23 | Wiring | **Host profile bridge** — `ApplicationEnvironmentProfile.tool_invocation_mode` → `RuntimeConfig` | **Done** | P1 | `environment_profile.py`, `catalog_runtime_bridge.py` | `test_catalog_runtime_bridge_tool_invocation.py` |
| TOOL-ENG-24 | Plugins | **Entry-point registry `intergrax.tool_invocation_patterns`** — custom host/agent patterns | **Done** | P2 | `tool_invocation_registry.py`, `config.py` | `test_tool_invocation_registry.py` |
| TOOL-ENG-27 | Observability | **Pattern trace** — `pattern_id`, `stop_reason`, `ops:tool_invocation_pattern` in `ToolsSummaryDiagV1` | **Done** | P2 | `plan_context_invocation.py`, `tools_summary.py` | `test_tools_pattern_telemetry.py` |
| TOOL-ENG-28 | CI | **`check_tool_invocation_patterns.py`** — shipped patterns registered; factory delegation | **Done** | P2 | `scripts/check_tool_invocation_patterns.py` | Gate script green |
| TOOL-ENG-30 | DX | **`lab_application` reference wiring** — `LAB_TOOL_INVOCATION_MODE` for each shipped mode | **Done** | P2 | `lab_application/host/settings.py` | `test_lab_tool_invocation_mode.py` |

#### Loop, governance, adaptive

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TOOL-ENG-6 | Loop | **Bounded tool loop** — `max_iterations`, native `role=tool`; refactored to `BoundedReactPattern` (TOOL-ENG-18) | **Done** | P1 | `tool_loop_step.py`, ADR-TOOL-002 | `test_tool_loop_integration.py` |
| TOOL-ENG-7 | Verify | **Post-tool verify** — `risk_level >= HIGH` → trace + enforce block | **Done** | P2 | `tool_verify_hooks.py`, `tool_loop.py` | `test_tool_verify_enforcement.py` |
| TOOL-ENG-8 | Governance | **`tools_mode=required` hard fail** | **Done** | P2 | `plan_context_invocation.py` | `ToolsRequiredError` when zero traces |
| TOOL-ENG-12 | Config | **`tool_choice` wiring** from host / `tools_mode` | **Done** | P2 | `tool_planning_policy.py`, patterns | `required` → tool_choice required |
| TOOL-ENG-10 | AHI | **Dynamic selection + invocation mode** — AHI hook + per-run resolver | **Done** | P3 | `tool_engine_wiring.py`, `adaptive_tool_mode_resolver.py` | `test_adaptive_tool_mode_resolver.py` · `check_tool_engine_ahi_hook.py` |

#### Documentation

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TOOL-ENG-DOC.4 | Docs | **Selection modes canon** — standard / semantic / hierarchical | **Done** | P1 | `architecture/TOOLS.md`, FLOW §15 | §modes live |
| TOOL-ENG-DOC.5 | Docs | **Invocation patterns canon** | **Done** | P1 | `architecture/TOOLS.md` §patterns | §patterns live |
| TOOL-ENG-DOC.6 | Docs | **Graph vs tool-pattern boundary** — ORCHESTRATION §50.4 + FLOW §15.1 cross-refs | **Done** | P2 | `ORCHESTRATION.md`, `NEXUS_EXECUTION_FLOW.md` | Linked from both domain pairs |
| TOOL-ENG-DOC.7 | Docs | **Selection plugin model canon** — standard / semantic / hierarchical + custom strategy surfaces | **Done** | P1 | `architecture/TOOLS.md` §selection plugin | §plugin model live |

**Delivery rule:** One **TOOL-ENG-\*** ID per PR → update this table + [§6.1e](#61e-harness-implementation-queue--tool-engine-active) + architecture [gap register](../architecture/TOOLS.md#engine-gap-register-canon) → `pytest -m gate` + new acceptance tests green.

**Suggested PR order (2026-06-12):**

```text
# Closed
TOOL-ENG-0 → 3 → 1 → 2 → 4 → 11 → 5 → DOC.4 → 6 → DOC.5

# Next — invocation pattern foundation (P0/P1)
TOOL-ENG-16 (ADR-TOOL-003) → 17 → 18 → 21 → 22 → 23 → 29 → 9 → 13 → 25

# Selection plugin + scale
TOOL-ENG-31 → 26 → 13 → 14 → 15 → 32

# Chain + plugins + observability
TOOL-ENG-20 → 24 → 27 → 28 → 30

# Governance closeout
TOOL-ENG-12 → 8 → 7 → 10
```

**Explicitly excluded:** New catalog bundles (§6.3), business agent tools (Phase K), replacing `ToolContract` / provider handlers.

### TOOL-ENG — Production readiness targets

| Metric | Baseline (2026-06-10) | Current (2026-06-12) | TOOL-ENG closeout target |
|--------|----------------------|----------------------|--------------------------|
| `tool_planner` wired on default hosts | No | **Done** | Auto from env + LLM |
| Arbitrary `tool_id` via `ToolRequest` | Partial | **Done** | Any registered id |
| `tools_context_scope` | Unused | **Done** | All 3 scopes |
| `EnginePlan.tool_ids` execution | Partial | **Done** | All listed ids |
| Planner schema vs `allowed_tools` | Full registry leak | **Done** | Filtered |
| Selection modes canon | Missing | **Done** (DOC.4) | Architecture + plan pair |
| Invocation patterns canon | Missing | **Done** (DOC.5) | §patterns + plan register |
| `ToolInvocationPattern` plugin | N/A | **Done** | Protocol + all shipped patterns (TOOL-ENG-16–30) |
| Parallel read-only invoke | **Done** | **Done** | `ParallelBatchPattern` (TOOL-ENG-9) |
| Deterministic tool chains | Agent code only | **Done** | `ToolChainSpec` + `DeterministicChainPattern` (TOOL-ENG-20) |
| Semantic tool index | Keyword overlap | **Done** | Vector top-k (TOOL-ENG-13) |
| Hierarchical category traversal | Metadata only | **Done** v1 | Multi-pass (TOOL-ENG-14) |
| Tool loop iterations | 1 | **Done** (`max_tool_iterations`) | Refactored to `BoundedReactPattern` (TOOL-ENG-18) |
| `tools_mode=required` | Warning trace | **Done** | Run failure (TOOL-ENG-8) |
| Invoker scope enforcement | Unwired | **Done** | Production wiring |
| Graph vs tool orchestration boundary | Implicit | **Documented** (DOC.5/6) | ORCHESTRATION + FLOW cross-refs |
| E2E multi-tool loop test | None | **Done** | `test_tool_loop_integration.py` |
| Custom invocation pattern from host | N/A | **Done** | Entry points (TOOL-ENG-24) |

### TOOL-ENG — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-10 | TOOL-ENG (register) | Full-stack audit; architecture + plan pair updated; Band 2ba queue opened |
| 2026-06-10 | TOOL-ENG-0,11,12 | Audit pass 2: planner bootstrap gap, dead tools_context_scope, tool_choice unwired |
| 2026-06-10 | TOOL-ENG-0 | `wire_catalog_tool_planner_if_enabled` in `RuntimeContext.build`; `tool_planner_prompt_id` via catalog bridge — **no ADR needed** (wiring only) |
| 2026-06-10 | TOOL-ENG-3 | `tool_scope_policy` → `RuntimeToolInvoker` in `RuntimeContext.build` — **no ADR needed** (wiring only) |
| 2026-06-10 | TOOL-ENG-1,2 | `catalog_dispatch.py` — plan per-id invoke + full-catalog gateway; ADR-TOOL-001 |
| 2026-06-10 | TOOL-ENG-4 | `allowed_tool_ids` on `ToolPlanningService`; state + step params wiring — **no ADR needed** |
| 2026-06-10 | TOOL-ENG-11 | `resolve_tool_planner_input` for `tools_context_scope` — **no ADR needed** |
| 2026-06-10 | TOOL-ENG-5 | `ToolSelectionStrategy` + lab `skill_pack` mode — **no ADR needed** |
| 2026-06-11 | TOOL-ENG-DOC.4 | Selection modes audit — standard / semantic / hierarchical canon in architecture pair; TOOL-ENG-13/14/15 registered |
| 2026-06-12 | TOOL-ENG-DOC.5,6 | Invocation-pattern audit — §patterns canon; ORCHESTRATION §50.4 + FLOW §15.1 boundary; TOOL-ENG-16–30,26,29 registered |
| 2026-06-12 | TOOL-ENG-16–18,21–23, TEST.1 | Layer completion S0–S3 — `ToolInvocationPattern` protocol, shipped patterns, config/bridge; ADR-TOOL-003 |
| 2026-06-12 | TOOL-ENG-9,29 | Layer completion S4 — `ParallelBatchPattern`, `ToolInvocationAggregate`, `max_parallel_tool_calls` |

---

### 6.1e Harness implementation queue — tool engine (closed 2026-06-12)

**Purpose:** Single ordered list for **Phase TOOL-ENG** (Band 2ba). **Opened 2026-06-10** · extended **2026-06-12** (invocation patterns).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TOOL-ENG-0** | Code | **Done** | `CatalogToolPlanner` bootstrap | `test_tool_engine_bootstrap.py` |
| 2 | **TOOL-ENG-3** | Code | **Done** | `tool_scope_policy` → invoker | scope deny on invoke |
| 3 | **TOOL-ENG-1** | Code + ADR | **Done** | Per-`tool_id` dispatch | `test_tool_runtime_catalog_dispatch.py` |
| 4 | **TOOL-ENG-2** | Code + ADR | **Done** | Full-catalog gateway | `test_tool_gateway.py` |
| 5 | **TOOL-ENG-4** | Code | **Done** | Plan constraints → planner | `test_tool_planning_constraints.py` |
| 6 | **TOOL-ENG-11** | Code | **Done** | `tools_context_scope` | scope enum tests |
| 7 | **TOOL-ENG-5** | Code | **Done** | `ToolSelectionStrategy` | `test_tool_selection_strategy.py` |
| 8 | **TOOL-ENG-DOC.4** | Docs | **Done** | Selection modes canon | architecture §modes |
| 9 | **TOOL-ENG-6** | Code + ADR | **Done** | Bounded tool loop → `BoundedReactPattern` | `test_tool_loop_integration.py` |
| 10 | **TOOL-ENG-DOC.5** | Docs | **Done** | Invocation patterns canon | architecture §patterns |
| 11 | **TOOL-ENG-16** | Code + ADR | **Done** | `ToolInvocationPattern` Protocol | ADR-TOOL-003 · conformance test |
| 12 | **TOOL-ENG-17** | Code | **Done** | `SinglePassPattern` | single-pass parity test |
| 13 | **TOOL-ENG-18** | Code | **Done** | `BoundedReactPattern` refactor | loop integration green |
| 14 | **TOOL-ENG-21** | Code | **Done** | `tool_invocation_pattern` config + factory | factory unit test |
| 15 | **TOOL-ENG-22** | Code | **Done** | `run_bounded_tool_loop` / `ctx.invoke_tool` → pattern delegation | existing tests green |
| 16 | **TOOL-ENG-23** | Code | **Done** | Host profile bridge | `test_catalog_runtime_bridge_tool_invocation.py` |
| 17 | **TOOL-ENG-29** | Code | **Done** | `ToolInvocationAggregate` | aggregate unit test |
| 18 | **TOOL-ENG-9** | Code | **Done** | `ParallelBatchPattern` | `test_parallel_batch_pattern.py` |
| 19 | **TOOL-ENG-13** | Code + ADR | **Done** | Semantic tool vector index | embedding rank gate |
| 20 | **TOOL-ENG-25** | Code | **Done** | `ParallelSemanticBatchPattern` | composite integration test |
| 21 | **TOOL-ENG-14** | Code + ADR | **Done** | Hierarchical selection | 2-pass integration |
| 22 | **TOOL-ENG-15** | Code + Docs | **Done** | `keyword_top_k` alias | alias in bridge |
| 23 | **TOOL-ENG-31** | Code | **Done** | `tool_selection_strategy` config inject | override factory test |
| 24 | **TOOL-ENG-26** | Code + ADR | **Done** | Selection strategy entry points | ADR-TOOL-004 · custom EP test |
| 25 | **TOOL-ENG-20** | Code | **Done** | `DeterministicChainPattern` + `ToolChainSpec` | chain integration test |
| 26 | **TOOL-ENG-24** | Code | **Done** | Invocation pattern entry points | custom pattern EP test |
| 27 | **TOOL-ENG-32** | Code | **Done** | Selection trace telemetry | ops:tool_selection test |
| 28 | **TOOL-ENG-27** | Code | **Done** | Pattern trace telemetry | diag payload test |
| 29 | **TOOL-ENG-28** | CI | **Done** | `check_tool_invocation_patterns.py` | gate green |
| 30 | **TOOL-ENG-30** | DX | **Done** | `lab_application` pattern examples | smoke per mode |
| 31 | **TOOL-ENG-DOC.6** | Docs | **Done** | ORCHESTRATION §50.4 + FLOW §15.1 boundary | linked pairs |
| 32 | **TOOL-ENG-DOC.7** | Docs | **Done** | Selection plugin model canon | §selection plugin |
| 33 | **TOOL-ENG-12** | Code | **Done** | `tool_choice` wiring | required mode test |
| 34 | **TOOL-ENG-8** | Code | **Done** | `tools_mode=required` fail | unit test |
| 35 | **TOOL-ENG-7** | Code | **Done** | Post-tool verify HIGH+ | middleware test |
| 36 | **TOOL-ENG-10** | Code | **Done** | AHI dynamic mode hook | fixture test |

**Explicitly excluded:** K.1, K.2, new product catalog tools — [§6.3 product backlog](../plan/PLATFORM_FOUNDATION.md).

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (catalog layer) — engine gaps tracked in **Phase TOOL-ENG** (2026-06-10)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-11.1 | §11 Tools | Sandboxed execution for code / side-effectful tools | P1 | **Done** |
| AUDIT-IDEAL-11.2 | §11 Tools | MCP / function-schema export for shipped tool catalog | P2 | **Done** |
| AUDIT-IDEAL-11.3 | §11 Tools | Oversized-tool lint enforcement in CI (adoption sweep) | P2 | **Done** |

**Follow-on (engine, not AUDIT-IDEAL id):** TOOL-ENG-1–10 — see [Phase TOOL-ENG](#phase-tool-eng--tool-engine-hardening-2026-06-10-audit).

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### 6.1c Harness implementation queue — tools/skills closeout (closed)

**Purpose:** Single ordered list for **Phase TS** (Band 2k). **Closed 2026-06-02** — all TS rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TS-DOC.1–2** | Docs | **Done** | Appendix J + cross-refs | Author map complete |
| 2 | **TS-1** | Code | **Done** | `catalog_runtime_bridge` + `RuntimeConfig.skill_profile` | `test_catalog_runtime_bridge.py` |
| 3 | **TS-2** | Code | **Done** | Harness host `resolve_llm_adapter` wiring | `test_harness_host_runtime_llm.py` |
| 4 | **TS-3** | Code | **Done** | `SkillResolverProtocol` | skill resolver tests green |

**Suggested PR order (complete):** TS-1 → TS-2 → TS-3 → TS-DOC.*.

**Explicitly excluded:** K.1, K.2, new product tools/skills, business agent packs — [§6.3a](#63a-business-backlog-register-consolidated).### 6.1aa Harness implementation queue — memory platform (closed)

**Purpose:** Phase MEM execution queue — **closed 2026-06-02** (48/48 Done). Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-1.1–MEM-1.4** | Code | **Done** | H-APP `MemoryProfile` + `ContextProfile.budget` + SQLite session → `RuntimeConfig` | MEM-1.5 gate test green |
| 2 | **MEM-2.1–MEM-2.3** | Code | **Done** | `SQLiteUserProfileStore` + bundle wiring + unit tests | LTM survives restart on sqlite profile |
| 3 | **MEM-1.6** | Docs/status | **Done** | H-APP.4.3 → **Done** | Bridge complete |
| 4 | **MEM-4.1–MEM-4.3** | Test | **Done** | Session + LTM + full-stack memory gates | acceptance/integration green |
| 5 | **MEM-5.1–MEM-5.2** | Test/Docs | **Done** | `engine_history_layer` tests + compression docs | unit + guide |
| 6 | **MEM-3.1–MEM-3.3** | Code | **Done** | Memory store plugin EP + reference fixture | bootstrap + gate |
| 7 | **MEM-0.3–MEM-DOC.*** | Docs | **Done** | Author cookbooks + Appendix G sync | guide updated |
| 8 | **MEM-6.*–MEM-7.*** | Code | **Done** | Retention enforcement + memory hooks | P2 after P0/P1 |
| 9 | **MEM-8.*–MEM-9.*** | RFC | **Done (RFC)** | Product memory layer + entity graph design | §6.3 gate for implementation |

**Suggested PR order:** See [Phase MEM — Suggested PR order](#mem--paydown-log).

**Explicitly excluded:** K.1, K.2, Mem0 SaaS product, entity graph ship (RFC only), business agent memory.

---

## Phase LEG — Legacy tool plan boolean closeout

**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (LEG-1–2); gate **612 passed**

**Audit basis:** Phase O.5a residual; `check_legacy_tool_plan_booleans.py`; Appendix J §J.6.

**Priority ladder:** **Band 2o** (§4.0) — closed; default queue = **§6.1** maintenance.

### LEG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| LEG-1 | LEG1 | **`tool_invocation_plan_from_capability_payload`** — gateway maps booleans → `tool_ids` without `from_legacy` | **Done** | `tool_runtime.py`, `tool_gateway.py` | `test_capability_payload_tool_plan.py` |
| LEG-2 | LEG2 | **Engine planner `tool_ids`** — parser populates `EnginePlan.tool_ids`; schema optional `tool_ids` | **Done** | `engine_planner_parse.py`, `nexus_llm_plan_builder.py` | `test_engine_plan_json_parser.py` |
| LEG-3 | LEG3 | **`plan_from_like` canonical path** — `from_tool_ids` only; `tool_gateway` removed from audit grandfather | **Done** | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | audit script green |

**Residual:** `ToolInvocationPlan.from_legacy()` retained in `tool_runtime.py` for explicit deprecation tests only; `EnginePlan.use_rag`/`use_websearch` remain on LLM schema for backward-compatible planner output.

---

---

## Phase TS — Tools & skills control plane closeout

**Status:** **Done** (2026-06-02) — **5/5** deliverables Done (TS-DOC.* + TS-1–3); gate **589 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11–§12; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix J**.

**Priority ladder:** **Band 2k** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bc](#62bc-phase-ts-execution-order-band-2k--closed) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

**Delivery rule:** One **TS-*** ID per PR → update master table + §6.1c + paydown log below → `pytest -m gate` + §6.1 scripts green.

### TS — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TS-DOC.1 | TS0 | **Appendix J** — tools & skills control plane map (§J.1–J.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| TS-DOC.2 | TS0 | **Cross-ref sync** — plan, README, AUDIT_MAP §11–§12, audit prompt ref #7 | **Done** | Medium | `docs/*` | Links resolve |
| TS-1 | TS1 | **`catalog_runtime_bridge.py`** — `tool_profile` / `skill_profile` on `RuntimeConfig` via `materialize_runtime_config` | **Done** | **Critical** | `catalog_runtime_bridge.py`, `runtime_config_bridge.py`, `config.py` | `test_catalog_runtime_bridge.py` |
| TS-2 | TS2 | **Harness host LLM wiring** — `resolve_llm_adapter(env)` → `build_nexus_loop_from_environment` | **Done** | High | `harness_host_runtime.py` | `test_harness_host_runtime_llm.py` |
| TS-3 | TS3 | **`SkillResolverProtocol`** — typed contract for skill composition resolution | **Done** | Medium | `skills/resolver.py`, `contract_resolution.py` | existing skill resolver tests green |

**Residual (not TS scope — track separately):** legacy `use_rag`/`use_websearch` booleans in `engine_planner` / `tool_gateway` (deprecation warnings; `check_legacy_tool_plan_booleans.py`).

### TS — Paydown log

| Date | TS ID | Summary |
|------|-------|---------|
| 2026-06-02 | TS-DOC.1, TS-DOC.2 | Appendix J + cross-refs; AUDIT_MAP §11–§12 authoring map |
| 2026-06-02 | TS-1, TS-2, TS-3 | Catalog runtime bridge, harness LLM wiring, SkillResolverProtocol; gate **589** |

**Phase TS complete when:** TS-1–3 + TS-DOC.* **Done**; §6.1c queue closed; Appendix J has no “planned wiring” gaps; gate **589** green. **Status: complete (2026-06-02).**

---

## Phase TOOL-ENG-DOC — Tool engine documentation canon (Band 2ar / 2bb)

**Status:** **Done** (2026-06-12) — **7/7** DOC rows · pipeline · selection modes · invocation patterns · selection plugin · graph boundary  
**Prerequisites:** Phase TS **Done** · Phase O **Done** · Phase LEG **Done**  
**Goal:** Canon in [`architecture/TOOLS.md`](../architecture/TOOLS.md) for selection (L6), orchestration (2a), atomic invoke (2b), logging — plus plugin extensibility  
**ADR:** **No ADR needed** for DOC rows; implementation rows TOOL-ENG-13/14/16/26 require ADR at code merge

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| TOOL-ENG-DOC.1 | **Tool execution pipeline** — diagram + phase table + entry paths | **Done** | Critical | `architecture/TOOLS.md` | select → orchestrate → invoke → log |
| TOOL-ENG-DOC.2 | **Component naming** — Tool engine vs `ToolRuntime` | **Done** | High | same | §Tool engine table |
| TOOL-ENG-DOC.3 | **Cross-ref sync** — FLOW §15, AUDIT_MAP §11, Appendix J | **Done** | Medium | `docs/*` | Links resolve |
| TOOL-ENG-DOC.4 | **Selection modes** — standard / semantic / hierarchical | **Done** | Critical | `architecture/TOOLS.md`, FLOW §15 | §modes |
| TOOL-ENG-DOC.5 | **Invocation patterns** — single / parallel / ReAct / chain / graph boundary | **Done** | Critical | `architecture/TOOLS.md`, FLOW §15.1, ORCH §50.4 | §patterns |
| TOOL-ENG-DOC.6 | **Graph vs tool-pattern boundary** | **Done** | High | `ORCHESTRATION.md`, `NEXUS_EXECUTION_FLOW.md` | §50.4 + §15.1 |
| TOOL-ENG-DOC.7 | **Selection plugin model** — `ToolSelectionStrategy`, surfaces A/B/C | **Done** | Critical | `architecture/TOOLS.md` | §selection plugin |

### TOOL-ENG-DOC traceability

| Pipeline phase | Canon section | Runtime modules |
|----------------|---------------|-----------------|
| Selection L6 | §[modes](../architecture/TOOLS.md#tool-selection-modes-production-strategies) · §[plugin model](../architecture/TOOLS.md#tool-selection-plugin-model-l6-extensibility) · FLOW §15 | `ToolSelectionStrategy`, `resolve_planner_allowed_tool_ids` |
| Planning L6b | §Multi-tool execution · §patterns | `ToolPlanningService`, `ToolPlannerProtocol` |
| Orchestration 2a | §[Invocation patterns](../architecture/TOOLS.md#tool-invocation-patterns-production-orchestration) · FLOW §15.1 | `ToolInvocationPattern` **Done** (TOOL-ENG-16), `run_bounded_tool_loop` / `resolve_invocation_pattern()` |
| Atomic invoke 2b | §pipeline · §42.12 gateway | `RuntimeToolInvoker`, `ToolRuntime` |
| Logging | §pipeline · FLOW §17 · OBS | `trace_event`, `TOOL_*`, `run_bounded_tool_loop` / `ctx.invoke_tool` |
| Gaps | §[Engine gap register](../architecture/TOOLS.md#engine-gap-register-canon) | Phase **TOOL-ENG** master register |

### TOOL-ENG-DOC — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-08 | TOOL-ENG-DOC.1–3 | Tool execution pipeline §; cross-refs |
| 2026-06-11 | TOOL-ENG-DOC.4 | Selection modes canon; TOOL-ENG-13/14/15 |
| 2026-06-12 | TOOL-ENG-DOC.5–7 | Invocation patterns + selection plugin + ORCH/FLOW boundary |

---

### 6.1d Harness implementation queue — tool engine docs (closed)

**Purpose:** Phase **TOOL-ENG-DOC** (Band 2ar) documentation closeout. **Closed 2026-06-08**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 1 | **TOOL-ENG-DOC.1–2** | Docs | **Done** | Architecture pipeline § + naming | Select / invoke / log covered |
| 2 | **TOOL-ENG-DOC.3** | Docs | **Done** | Cross-ref sync | FLOW, AUDIT_MAP, Appendix J |

---

---

### Phase O — Tool Library & Unified Tool Model (Tier-0)

**Canon:** §7.1.6–§7.1.7, §22, §42.12  
**Goal:** Ship a reusable **Tool Library** catalog (mirror Integration Library) and migrate legacy pipeline flags (`use_rag`, `use_websearch`) to explicit catalog tools.

**Prerequisite:** Phase M.3 (`IntegrationProfile`) available; tool engine (`ToolRegistry`, `RuntimeToolInvoker`) exists.

**Catalog reference:** [`architecture/TOOLS.md`](architecture/TOOLS.md)

**Delivery rule:** One domain or migration slice per iteration — implement → gate → update `architecture/TOOLS.md` → next step.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| O.0 | Architecture & catalog documented | **Done** | §7.1.6–§7.1.7, §22 | Runtime canon + `architecture/TOOLS.md` + this section (2026-05-30) |
| O.1 | Extended `ToolContract` | **Done** | §22 | `ToolRiskLevel`, `ToolRetryPolicy`, metadata fields; invoker timeout/retry/trace (2026-05-30) |
| O.2 | `ToolCatalog` + `ToolProfile` + `ToolWiringContext` | **Done** | §7.1.6 | `intergrax/tools/registry/`; `build_registry_from_profile`; RuntimeConfig wiring (2026-05-30) |
| O.3 | Context tools: `rag.retrieve`, `websearch.query` | **Done** | §7.1.7, §22.1 | `providers/rag/`, `providers/websearch/` (2026-05-30) |
| O.4 | Reference domain: `jira.*` tools | **Done** | §7.1.6 | `get_issue`, `add_comment`, `search_tasks` over `IssueTracker` (2026-05-30) |
| O.4b | Catalog domain bundles: `confluence.*`, `notify.send`, observability, `sandbox.exec` | **Done** | §7.1.6 | All first-party catalog tools registered (2026-05-30) |
| O.5 | **Unified tool model migration** | **Done** | §7.1.7, §22.2 | `tool_ids` on plans; RagStep/WebsearchStep → catalog shims (2026-05-30) |
| O.6 | Schema exporters (OpenAI + MCP) | **Done** | §7.1.6 | `tools/exporters/`; MCP catalog mount on lab/poc_template (2026-05-30) |
| O.7 | Migrate legacy `ToolBase` → `ToolContract` | **Done** | §5.2.2 | `ChatAgent` → registry; `tools_base` deprecated (2026-05-30) |
| O.8 | `ToolProfile` in Tier-3 scaffold | **Done** | §7.4.8 | `tool_wiring.py` template; lab + poc_template reference (2026-05-30) |
| O.9 | Agent Creation Guide Appendix E update | **Done** | — | Unified model + ToolProfile examples (2026-05-30) |
| O.10 | Gate tests for catalog conformance | **Done** | — | `tests/unit/tools/providers/` — all catalog bundles (2026-05-30) |
| O.11 | Phase P wave 2 context tools: `websearch.read_url`, `confluence.search` | **Done** | §7.1.7, §22.1 | `providers/websearch/read_url_*`, confluence alias (2026-05-30) |
| O.12 | Phase P wave 3 tools: `websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces` | **Done** | §7.1.7, §22.1 | Extended `ObservabilityBackend.query_traces`, vector `list_collections` (2026-05-30) |

#### O — Step-by-step implementation sequence

Execute **strictly in order** for foundation (O.1–O.4); O.5–O.10 may overlap after O.4 reference tools land.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | O.1 | Extend `ToolContract` + update `RuntimeToolInvoker` for new fields | Unit tests pass; backward compatible defaults |
| 2 | O.2 | Add `tools/registry/catalog.py`, `profile.py`, `ToolWiringContext` dataclass | `register_default_tools()` no-op registry; profile enables subset |
| 3 | O.3 | Implement `providers/rag/` and `providers/websearch/` handlers | **Done** — `rag.retrieve`, `websearch.query` + tests |
| 4 | O.4 | Implement `providers/jira/` bundle (3 tools) | **Done** — conformance tests with mocked `IssueTracker` |
| 4b | O.4b | Implement remaining catalog bundles (`confluence`, `notify`, `observability`, `sandbox`) | **Done** — all tool_ids in `register_default_tools()` |
| 5 | O.5a | Add `tool_ids` to plan models; map legacy booleans → tool_ids | **Done** — `ToolInvocationPlan`, `LegalToolPlan` |
| 6 | O.5b | `rag.retrieve` (catalog) / `websearch.query` (catalog) delegate to catalog tools | **Done** — `catalog_context.py` shim |
| 7 | O.5c | Update `LegalToolPlan` / engine plans to tool list | **Done** — bridge passes `tool_ids` |
| 8 | O.6 | MCP + OpenAI exporters from single catalog | **Done** — `tools/exporters/` |
| 9 | O.7 | Remove `ToolBase` usage from production paths | **Done** — `ChatAgent` uses registry `ToolRegistry` |
| 10 | O.8–O.10 | Scaffold, docs, gate | **Done** |

#### O.4 — Adding a new tool provider (checklist)

Copy into every `tools/providers/<domain>/USAGE.md`:

```text
[ ] 1. Define Input/Output Pydantic models (LLM-friendly field names)
[ ] 2. Implement ToolHandler — compose integration contract(s), no vendor SDK
[ ] 3. Build ToolContract per tool (description tuned for model selection)
[ ] 4. register_<domain>_tools(registry, ctx: ToolWiringContext)
[ ] 5. Register in tools/registry/catalog.py
[ ] 6. Unit tests with fakes (no live vendor in default gate)
[ ] 7. Wire in lab or poc_template via ToolProfile
[ ] 8. Update architecture/TOOLS.md status + this plan tracker
```

#### T-EXPAND — Integration bridge catalog expansion (2026-06-07) — **Done**

**Goal:** Close the integration→tool coverage gap (~78% integrations without LLM tools) by shipping provider-agnostic bundles that compose existing `IntegrationCategory` contracts.

| Wave | Bundles | Tools | Status |
|------|---------|------:|--------|
| T1 (DX / runtime-bound) | `workspace`, `memory`, `knowledge`, `document`, `browser`, `storage` (get) | 12 | **Done** |
| T2 (prod harness) | `storage` (+put/presigned/delete), `issues`, `platform` | 10 | **Done** |
| T3 (async / graph / collab / cache) | `message_bus`, `graph`, `collaboration`, `cache` | 8 | **Done** |

**Delivered:**

- **67** catalog `tool_id` values · **28** shipped bundles (`shipped_plugins.py`)
- Typed `ToolWiringContext` slots for all new integration categories
- `TaskMemoryViewBinding` protocol (avoids Tier-0 ↔ UAEP import cycle)
- UAEP `runtime_bound_catalog.py` for `workspace.*` / `memory.*` (mirrors `sandbox.exec`)
- `extend_tool_profile_for_integration()` P6 auto-enable (excludes ingest-only `document_parser`)
- Gate: **909** passed (`uv run pytest -m gate -q`)

**Follow-up (2026-06-07) — Done:**

- `IssueCreator` protocol + `issues.create_issue` (no `getattr` in GitLab tool path)
- `harness.integration_bridge_smoke` skill pack + resolver test fix (skills vs tools `build_registry_from_profile`)
- Lab harness `wire_lab_tools(harness=True)` enables runtime-bound + bridge tools
- PoC template `extend_tool_profile_for_integration()` wiring
- MCP full-catalog export smoke (130 tools)

#### T-EXPAND T4 — Agent Builder Essentials (2026-06-07) — **Done**

**Goal:** Close highest-ROI integration→tool gaps for agent/environment builders (SQL, document JSON, RAG lifecycle, workspace DX, collaboration read path, auto-enable wiring).

| Bundle | Tools | Status |
|--------|------:|--------|
| `database` | `database.query`, `database.execute` | **Done** |
| `records` | `records.get`, `records.put`, `records.delete`, `records.query` | **Done** |
| `rag` (+2) | `rag.delete_documents`, `rag.describe_collection` | **Done** |
| `workspace` (+2) | `workspace.delete_file`, `workspace.search` | **Done** |
| `collaboration` (+4) | `collaboration.list_messages`, `get_message`, `list_calendar`, `get_user` | **Done** |
| wiring | `relational_store` / `document_store` ctx slots; auto-enable notify/obs/database/records/collaboration | **Done** |

**Delivered:** **81** catalog `tool_id` values · **30** shipped bundles.

#### T-EXPAND T5 — Production Harness Ops (2026-06-07) — **Done**

**Goal:** Production harness operations for identity, persisted run trace read, integration health probes, online evaluation registry, and platform/security extensions.

| Bundle | Tools | Status |
|--------|------:|--------|
| `identity` | `identity.verify_token`, `identity.get_user`, `identity.list_tenants` | **Done** |
| `harness` | `harness.get_run`, `harness.list_runs`, `harness.get_run_cost`, `harness.get_run_events` | **Done** |
| `health` | `health.check_integration`, `health.check_profile` | **Done** |
| `eval` | `eval.record_observation`, `eval.list_observations`, `eval.summarize_release` | **Done** |
| `security` (+1) | `security.summarize_findings` | **Done** |
| `platform` (+1) | `platform.put_secret` | **Done** |
| wiring | `trace_reader` / `evaluation_registry` / `integration_profile` ctx slots; runtime-bound `harness.*`; observability bundle promoted STABLE | **Done** |

**Delivered:** **95** catalog `tool_id` values · **34** shipped bundles.

#### T-EXPAND T6 — LKW Filesystem + Harness Economics (2026-06-07) — **Done**

**Goal:** LKW read-only filesystem browse (LKW.3), V-COST/billing tool surface, rerank/cache/CRM/platform extensions.

| Bundle | Tools | Status |
|--------|------:|--------|
| `filesystem` | `filesystem.list`, `filesystem.glob`, `filesystem.read_text`, `filesystem.stat` | **Done** |
| `billing` | `billing.record_usage`, `billing.list_usage` | **Done** |
| `cost` | `cost.get_run_budget`, `cost.check_quota` | **Done** |
| `crm` | `crm.get_account`, `crm.list_contacts`, `crm.list_tickets` | **Done** |
| `platform` (+1) | `platform.delete_secret` | **Done** |
| `rag` (+1) | `rag.rerank` | **Done** |
| `cache` (+2) | `cache.delete`, `cache.list_keys` | **Done** |
| wiring | `read_allowlist_roots` ctx slot; runtime-bound `cost.*`; LKW auto-enable filesystem | **Done** |

**Delivered:** **110** catalog `tool_id` values · **38** shipped bundles.

#### T-EXPAND T7 — Index Lifecycle + Async Queue (2026-06-07) — **Done**

**Goal:** RAG index inspection, async task queue ops, observability range/tail, eval release compare, cost forecast.

| Bundle | Tools | Status |
|--------|------:|--------|
| `message_bus` (+2) | `message_bus.list_tasks`, `message_bus.cancel` | **Done** |
| `rag` (+3) | `rag.list_documents`, `rag.get_document`, `rag.check_index_status` | **Done** |
| `document` (+1) | `document.parse_preview` | **Done** |
| `observability` (+2) | `metrics.query_range`, `logs.tail` | **Done** |
| `eval` (+1) | `eval.compare_releases` | **Done** |
| `cost` (+1) | `cost.forecast_spend` | **Done** |
| contracts | `TaskQueue.cancel` / `list_tasks`; `VectorStoreDocumentListerBinding` | **Done** |
| wiring | auto-enable message_bus + observability extensions; runtime-bound `cost.forecast_spend` | **Done** |

**Delivered:** **120** catalog `tool_id` values · **38** shipped bundles.

#### T-EXPAND T8 — Governance + Agent Safety + LKW write (2026-06-07) — **Done**

**Goal:** Read-only HITL ops, allowlisted filesystem write, RAG metadata search/purge, schema introspection, CI/CD workflow ops.

| Bundle | Tools | Status |
|--------|------:|--------|
| `hitl` (+3, new) | `hitl.list_pending`, `hitl.get_decision`, `hitl.summarize_queue` | **Done** |
| `filesystem` (+1) | `filesystem.write_text` | **Done** |
| `rag` (+2) | `rag.search_by_metadata`, `rag.purge_collection` | **Done** |
| `database` (+1) | `database.describe_schema` | **Done** |
| `records` (+1) | `records.describe_collection` | **Done** |
| `platform` (+2) | `platform.list_workflow_runs`, `platform.cancel_workflow_run` | **Done** |
| contracts | `HumanDecisionStoreBinding`; `CiCdBackend.list/cancel`; `VectorstoreIndexLifecycleBinding.search/purge` | **Done** |
| wiring | LKW auto-enable write + RAG maintenance; integration profile CI/CD + schema tools | **Done** |

**Delivered:** **130** catalog `tool_id` values · **39** shipped bundles.

#### T-EXPAND T9 — Async orchestration + interaction (2026-06-07) — **Done**

**Goal:** Workflow run ops, notify batch, collaboration write-back, websearch cache invalidation, harness run diff/export, interaction session reads.

| Bundle | Tools | Status |
|--------|------:|--------|
| `workflow` (+2) | `workflow.list_runs`, `workflow.cancel_run` | **Done** |
| `notify` (+1) | `notify.send_batch` | **Done** |
| `collaboration` (+2) | `collaboration.reply_message`, `collaboration.create_event` | **Done** |
| `websearch` (+1) | `websearch.invalidate_cache` | **Done** |
| `harness` (+2) | `harness.compare_runs`, `harness.export_run_bundle` | **Done** |
| `interaction` (+2, new) | `interaction.list_sessions`, `interaction.get_last_input` | **Done** |
| contracts | `WorkflowOrchestratorBackend.list/cancel`; `CollaborationSuite.reply/create`; `WebSearchCacheBinding` | **Done** |
| wiring | integration profile workflow/collaboration/notify extensions; `session_storage` via `session_tool_wiring.py` + `SessionStorageToolBinding` | **Done** |

**Delivered:** **140** catalog `tool_id` values · **40** shipped bundles.

**Verification:** `152 passed` (`tests/unit/tools/providers/` + exporters) · `check_harness_no_getattr.py` OK · MCP full-catalog export smoke (**140** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{workflow,notify,collaboration,websearch,harness,interaction}/`

#### T-EXPAND T10 — LKW storage bridge + deferred scheduling (2026-06-07) — **Done**

**Goal:** Close T8/T9 deferred tools (`workspace.export_artifact`, `notify.schedule`) and extend builder/LKW ops without new bundles.

| Bundle | Tools | Status |
|--------|------:|--------|
| `workspace` (+2) | `workspace.export_artifact`, `workspace.import_artifact` | **Done** |
| `notify` (+1) | `notify.schedule` | **Done** |
| `interaction` (+1) | `interaction.get_session_history` | **Done** |
| `eval` (+1) | `eval.export_observations` | **Done** |
| `storage` (+1) | `storage.exists` | **Done** |
| `memory` (+1) | `memory.delete_key` | **Done** |
| `pagerduty` (+1) | `pagerduty.acknowledge_incident` | **Done** |
| `message_bus` (+1) | `message_bus.purge_completed` | **Done** |
| `records` (+1) | `records.count` | **Done** |
| contracts | `ScheduledNotificationBinding`; `SessionStorageBinding.get_session_history`; `TaskMemoryViewBinding.delete`; `TaskQueue.purge_completed` | **Done** |
| wiring | `notify_tool_wiring.py` + `PolicyScopedMemoryView.delete` | **Done** |

**Delivered:** **150** catalog `tool_id` values · **40** shipped bundles.

**Verification:** `164 passed` (`tests/unit/tools/providers/` + exporters) · `check_harness_no_getattr.py` OK · MCP full-catalog export smoke (**150** tools)

**Closeout notes (accepted platform limits):**

| Area | Platform behavior | Product follow-up |
|------|-------------------|-------------------|
| `notify.schedule` | Records deferred delivery in `ScheduledNotificationBinding` (in-memory default via Tier-3 wiring) | Production dispatcher/cron in application host |
| `message_bus.purge_completed` | **Done** — KV task index on broker queues (`rabbitmq`, `kafka`); Celery unchanged | Residual: Celery result-backend purge |
| `pagerduty.acknowledge_incident` | **Done** — `PagerDutyEventsClient.acknowledge_incident` + adapter + typed `PagerDutyIncidentChannel` | — |

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{workspace,notify,interaction,eval,storage,memory,pagerduty,message_bus,records}/`

#### T-EXPAND T11 — HITL write path + cloud/vector store ops (2026-06-07) — **Done**

**Goal:** Close T8/T10 deferred governance and integration-bridge gaps without product scope.

| Bundle | Tools | Status |
|--------|------:|--------|
| `hitl` (+2) | `hitl.submit_response`, `hitl.list_for_task` | **Done** |
| `notify` (+2) | `notify.list_scheduled`, `notify.cancel_scheduled` | **Done** |
| `cloud_platform` (new) | `cloud_platform.health`, `cloud_platform.resolve` | **Done** |
| `vector_store` (new) | `vector_store.count`, `vector_store.delete`, `vector_store.list_collections`, `vector_store.health` | **Done** |
| contracts | `HumanDecisionStoreBinding.record` / `list_for_task`; `ScheduledNotificationBinding.cancel_scheduled` | **Done** |
| wiring | `ToolWiringContext.cloud_platform`; `IntegrationProfile` cloud platform resolution | **Done** |

**Delivered:** **160** catalog `tool_id` values · **42** shipped bundles.

**Verification:** provider unit tests + MCP full-catalog export smoke (**160** tools) · `check_harness_no_getattr.py` OK

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{hitl,notify,cloud_platform,vector_store,health}/`

#### T-EXPAND T12 — Integration slot health + notify dispatcher (2026-06-07) — **Done**

**Goal:** Close post-T11 harness ops gaps (category health probes, scheduled notify dispatch, Celery purge index).

| Bundle | Tools | Status |
|--------|------:|--------|
| `health` (+9) | `health.check_object_storage`, `health.check_key_value_cache`, `health.check_message_bus`, `health.check_graph_store`, `health.check_identity_provider`, `health.check_relational_store`, `health.check_wiki_knowledge`, `health.check_search_provider`, `health.check_notification_channel` | **Done** |
| `notify` (+1) | `notify.dispatch_due` | **Done** |
| queue | Celery optional KV task index + `purge_completed` | **Done** |
| contracts | `ScheduledNotificationBinding.mark_delivered` | **Done** |
| planner | LEG-DEPTH — remove `use_rag`/`use_websearch` from LLM schema; deprecation trace | **Done** |
| observability | OBS-DEPTH.2 trace bridge phase gate; live emit via `runtime_event_bus` | **Done** |

**Delivered:** **170** catalog `tool_id` values · **42** shipped bundles.

#### T-EXPAND T13 — CRIT-V eval tools (2026-06-07) — **Done**

**Goal:** Ship semantic verification tools for Phase CRIT-V (PEV verify depth) without Nexus orchestrator wiring.

| Bundle | Tools | Status |
|--------|------:|--------|
| `eval` (+2) | `eval.judge`, `eval.trajectory` | **Done** |

**Delivered:** **172** catalog `tool_id` values · **42** shipped bundles.

**Verification:** `test_eval_critic_tools.py` · `test_catalog_expansion.py` (172) · MCP export smoke (**172** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md)

#### T-EXPAND T14 — Agent Builder DX introspection (2026-06-08) — **Done**

**Goal:** Runtime/catalog introspection for agent builders — discover tools, agents, and skill resolution without reading source.

| Bundle | Tools | Status |
|--------|------:|--------|
| `catalog` | `catalog.list_tools`, `catalog.describe_tool` | **Done** |
| `agent` | `agent.list_agents`, `agent.get_contract` | **Done** |
| `skill` | `skill.resolve` | **Done** |

**Delivered:** **175** catalog `tool_id` values · **45** shipped bundles.

#### T-EXPAND T15 — Sandbox execution depth (2026-06-08) — **Done**

**Goal:** Close `SANDBOX_REQUIRED_TOOLS` policy gap (`code.exec`, `script.run`, `browser.run`) and sandbox self-discovery.

| Bundle | Tools | Status |
|--------|------:|--------|
| `sandbox` (+4) | `code.exec`, `script.run`, `browser.run`, `sandbox.list_operations` | **Done** |
| runtime | `AGENT_BUILDER_SANDBOX_OPERATIONS` + `run_python` / `run_script` / `browser_fetch` session ops | **Done** |

**Delivered:** **179** catalog `tool_id` values · **45** shipped bundles.

**ADR:** **No ADR needed** — extends existing sandbox session ops; policy constants already referenced in `sandbox_runtime.py`.

#### T-EXPAND T16 — Memory & context builder surface (2026-06-08) — **Done**

**Goal:** Agent-facing LTM, task memory search, and context budget helpers.

| Bundle | Tools | Status |
|--------|------:|--------|
| `ltm` (new) | `ltm.search`, `ltm.write_fact` | **Done** |
| `memory` (+1) | `memory.search` | **Done** |
| `context` (new) | `context.summarize`, `context.estimate_tokens` | **Done** |
| bindings | `UserProfileManagerBinding` on `ToolWiringContext` | **Done** |

**Delivered:** **184** catalog `tool_id` values · **47** shipped bundles.

#### T-EXPAND T17 — Integration completeness (2026-06-08) — **Done**

**Goal:** HTTP allowlist client, interaction reply, issue update, RAG preview dry-run.

| Bundle | Tools | Status |
|--------|------:|--------|
| `http` (new) | `http.request` | **Done** |
| `interaction` (+1) | `interaction.post_reply` | **Done** |
| `issues` (+1) | `issues.update_issue` | **Done** |
| `rag` (+1) | `rag.preview_retrieval` | **Done** |
| contracts | `HttpClientBackend`, `IssueUpdater`, `AllowlistHttpClient` | **Done** |

**Delivered:** **190** catalog `tool_id` values · **48** shipped bundles.

**Verification:** `test_t14_t17_builder_tools.py` · `test_catalog_expansion.py` (190) · MCP export smoke (**190** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{catalog,agent,skill_tool,ltm,context_tool,http}/`


**Problem (Phase O):** Two parallel mechanisms — boolean plan flags dispatching pipeline steps vs `ToolRegistry` for function tools.

**Phase O outcome:** Unified **contracts** (`tool_ids` on plans, catalog shims for rag/websearch). **Phase TOOL-ENG** closes runtime **dispatch** and **gateway** gaps.

### Dispatch state — actual vs target

```text
LEGACY (deprecated, still mapped):
  plan.use_rag=True        → RagStep → catalog_context → rag.retrieve
  plan.use_websearch=True  → WebsearchStep → catalog_context → websearch.query
  plan.use_tools=True      → ToolsStep → ToolPlanningService → RuntimeToolInvoker

ACTUAL (2026-06-10):
  plan.tool_ids=["rag.retrieve", "websearch.query"]
      → normalized() sets use_rag / use_websearch → pipeline steps

  plan.tool_ids=["jira.search_tasks", "database.query"]
      → catalog_dispatch → RuntimeToolInvoker (TOOL-ENG-1 **Done**)
      → use_tools=True runs ToolsStep with planner allow-list from plan `tool_ids` (TOOL-ENG-4)

  ctx.invoke_tool(ToolRequest(tool_name="jira.get_issue"))
      → catalog_dispatch via RuntimeToolGateway (TOOL-ENG-2 **Done**)

TARGET (remaining TOOL-ENG):
  Multi-iteration tool loop (TOOL-ENG-6)
  Optional multi-iteration tool loop (TOOL-ENG-6)
```

**Compatibility (O.5a / LEG):** `ToolInvocationPlan.from_legacy(use_rag=…)` maps booleans to default tool_ids. Deprecation trace when legacy-only booleans used.

**Context injection:** `rag.retrieve` and `websearch.query` set `injects_context=true`; pipeline merges via `catalog_context` + `run_bounded_tool_loop` / `ctx.invoke_tool` system inject (§22.1).

**Configuration reference:** [`architecture/TOOLS.md`](../architecture/TOOLS.md) — [Runtime configuration reference](../architecture/TOOLS.md#runtime-configuration-reference), [Multi-tool execution](../architecture/TOOLS.md#multi-tool-execution-semantics), [§42.12 gateway](../architecture/TOOLS.md#4212-gateway-surface-toolrequest).

**Out of scope (TOOL-ENG):**

- Domain-specific tools inside `agents/` (Tier-2; register via `ToolProvider` if reusable)
- New integration categories (Phase M)
- Product-only tool packs (§6.3 / Phase K)

---

## Phase TOOLS-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates 2026-06-12 Layer Completion; no open P0/P1  
**Prerequisites:** TOOL-ENG **Closed** (36/36)  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| TOOLS-LC-S1 | **Re-audit** — TOOL-ENG register + tier-0/1 verdict | **Done** | High | No P0/P1 |
| TOOLS-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| TOOLS-LC-S3 | **Gate verification** | **Done** | High | 58 unit tests · 2 CI scripts |
| TOOLS-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** hierarchical LLM category pass · per-tool L1 critic (CVL) · host EP pattern packages

### 6.1av Harness implementation queue — Tools audit maintenance (planned)

**Source:** Layer 8 audit (2026-06-18) — `TOOLS` layer 11 · [`../audit_results/2026-06-18/TOOLS.md`](../audit_results/2026-06-18/TOOLS.md)  
**Priority ladder:** **Band 1** (§6.1) — selection depth + DX hygiene; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **TOOL-MAINT-01** | Code | P2 | **Done** | ADR-TOOL-005 v2 — optional LLM category pass in `hierarchical_tool_selector.py` | Integration test; deterministic default preserved |
| 2 | **TOOL-MAINT-02** | Docs/Trace | P2 | **Done** | Per-tool L1 critic output trace contract — cross-ref CVL; canon acceptance in TOOLS §critic hook | Trace payload documented; CVL gate references tool_id |
| 3 | **TOOL-MAINT-03** | DX | P3 | **Done** | Host EP pattern packages — scaffold/docs for custom entry-point tool patterns | Scaffold or guide section; example host wiring |
| 4 | **TOOL-MAINT-04** | DX | P3 | **Done** | Tool gate subset in `intergrax doctor` — `check_tool_injection_defense` + `check_legacy_tool_plan_booleans` | `intergrax doctor --ci` runs tool checks |

**Suggested PR order:** TOOL-MAINT-01 → TOOL-MAINT-02 → TOOL-MAINT-04 → TOOL-MAINT-03.

**Cross-domain (not TOOLS-owned):** PF-MAINT-LEG-01 — legacy `use_rag`/`use_websearch` planner schema — [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

---

*End of Tools Implementation Plan.*
