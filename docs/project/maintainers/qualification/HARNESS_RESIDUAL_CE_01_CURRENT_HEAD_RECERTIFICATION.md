# HARNESS-RESIDUAL / CE-01 — Current-HEAD Context Pipeline Ownership Convergence

**Status:** Cursor recertification — **BLOCKED** (not closure)  
**START_HEAD / AUDITED_HEAD:** `433ff5fb97262c4bad84ea2a4173dc3711d7aa08` (`development`)  
**Parent:** Residual Harness convergence before HARNESS-FINAL  
**Qualification package owner:** `tests/qualification/ce_01/` (CE-Q1..CE-Q15 catalog)

## Stage reconciliation

| Workstream | State |
| ---------- | ----- |
| HARNESS-W7 / HOST-01 | CLOSED @ `5d5442dae7769671a7532a973428c8203b492ec6` |
| HARNESS-W8 / BG-01 | CLOSED @ `3a9b58e6fa1e14ddc9786bd9a447a1d76552d196` |
| SCHED-01 | **CLOSED** (independently accepted @ `433ff5fb97262c4bad84ea2a4173dc3711d7aa08`) |
| **CE-01** | **CURRENT** (this record) |
| CE-02 | NOT ENTERED |
| HARNESS-FINAL | NOT ENTERED (blocked on CE-01 + residual) |

## Canonical ownership (§1)

```text
CONTEXT_ASSEMBLY_AUTHORITY = CONTEXT_ENGINEERING
```

Separate owners unchanged: UCL, Memory, RAG, Collaborative Work / MP-5 ContextView.

**Nexus:** implementation locus under `intergrax/runtime/nexus/context/**` — not a second semantic owner when consuming `ContextEngine` / `ContextAssemblyRequest` contracts from `intergrax/context/**`.

## Production composition owner (§10)

Exactly one sanctioned EE materialization root for host-backed runs:

```text
ApplicationEnvironmentProfile
  → application_environment_context_composition.py
      resolve_context_engine_from_environment
      resolve_context_orchestrator_from_environment (codebase preset only)
      resolve_context_manager_from_environment
  → host_orchestration_environment_spec_builder.py (resolved_context_manager)
  → materialize_host_orchestration_backend (NexusLoop context_manager=spec)
  → GraphExecutor(context_manager=shared manager)
```

Tier-3 re-export: `intergrax/applications/_shared/context_wiring.py` delegates to EE composition — not a second semantic root.

**Unsanctioned production composition roots (inventory §35):** 0 for certified harness host path (EBH-4-R1 gates: direct `NexusLoop()` outside EE = 0 in production scan).

## Closed-world model-call inventory (§5)

| Surface | Caller | Context owner | Budget owner | Provenance | Preflight | Classification |
| ------- | ------ | ------------- | ------------ | ---------- | --------- | -------------- |
| ACP step | Authoring + `build_acp_assembly_request` → `ContextEngine.assemble` | CE | CE `ContextBudgetSnapshot` / compiler | `AssembledContext.provenance` | CE pre-context policy + compile validation | **CANONICAL CE** |
| Graph agent node | `ContextManager.build_agent_context_async` → `engine.assemble` | CE | `_resolve_context_budget_policy` + engine | graph assembly provenance | CE pipeline | **CANONICAL CE** (when engine+adapter wired) |
| UAEP primary turn | `assemble_uaep_session_messages` | CE | CE via `build_graph_provider_context_bundle` runtime | assembled provenance | CE | **CANONICAL CE** |
| Bounded tool loop | `assemble_iterative_tool_planner_messages` | CE | same | tool blocks → fragments | CE | **CANONICAL CE** |
| Agent retry/replan | Same graph/task Nexus path | CE | shared manager policy | CE events | CE | **CANONICAL CE** |
| Child / delegation | `ExploreChildContextEngine` via composition | CE preset | CE | CE | CE | **LEGAL INTERNAL CE IMPLEMENTATION** |
| Codebase preset | `ContextOrchestrator` + `CodebaseContextEngine` | CE bounded preset | CE | CE | CE | **LEGAL INTERNAL CE IMPLEMENTATION** |
| ContextManager core fallback | `_build_agent_context_core` when `engine is None` or `llm_adapter is None` | CE presentation shim | local `ContextBudgetPolicy` trim only | partial | no full CE assemble | **LEGACY COMPATIBILITY — NON-PRODUCTION** on certified host path (manager always materialized with engine+adapter) |
| NexusLoop default `ContextManager()` | `nexus_loop.py` when `context_manager is None` | same fallback | same | partial | partial | **LEGACY COMPATIBILITY — NON-PRODUCTION** (lab/debug/tests; production uses `spec.context_manager`) |
| GraphExecutor default manager | `context_manager or ContextManager(...)` | fallback if caller omits | default chars | partial | partial | **LEGACY COMPATIBILITY — NON-PRODUCTION** when host injects manager (production materialization always injects) |
| Tool-selection LLM prompt | `hierarchical_tool_selector` (planner adjunct) | not model-facing agent turn | n/a | n/a | n/a | **OUTSIDE CURRENT FROZEN SCOPE — WITH EVIDENCE** (tool metadata selection, not certified primary agent context path) |
| Direct adapter in Tier-0 tests/fixtures | various | n/a | n/a | n/a | n/a | **LEGACY COMPATIBILITY — NON-PRODUCTION** |

**Unclassified certified surfaces:** 0.

## Context construction inventory — production (§35)

| Symbol | Production call sites | Classification |
| ------ | --------------------- | -------------- |
| `resolve_context_manager_from_environment` | `host_orchestration_environment_spec_builder.py` | **Sanctioned owner** |
| `resolve_context_engine_from_environment` | composition + graph delegation helper | **Sanctioned owner** |
| `ContextManager(` | composition only (production) | **A — CE semantics materialization** |
| `ContextManager(` | `nexus_loop.py`, `graph_executor.py` defaults | **B avoided in production** via injected manager |
| `DefaultNexusContextEngine(` | composition presets | **LEGAL INTERNAL** |
| `CodebaseContextEngine(` | `engine_preset=codebase` | **LEGAL INTERNAL** |
| `ContextOrchestrator(` | codebase preset only | **Bounded preset — not second runtime** |
| `ContextCompiler(` | inside `DefaultNexusContextEngine` | **Same semantic budget contract** |

## Ownership graph (§E)

```text
Memory / RAG / UCL / ContextView
  → typed fragments / plans / reads
  → CONTEXT_ENGINEERING (ContextEngine.assemble)
  → model-facing messages + provenance
  → Execution / Nexus (UAEP, GraphExecutor, ACP bridge) consumer
```

## Nexus vs CE (§F)

Nexus modules implement CE contracts; they do not define parallel `ContextAssemblyRequest` semantics. Tier-3 configures `ContextProfile`; it does not privately concatenate prompts on certified paths (CE-Q15 gate + application legacy RAG import scan).

## Budget / preflight matrix (§G)

Certified paths: adapter window → `ContextBudgetPolicy` / `ContextBudgetSnapshot` → allocation/degradation in `DefaultNexusContextEngine` → `ContextCompiler` → preflight validation. No independent graph-only or UCL-only model-input budget owner on wired paths.

## Direct assembly bypass (§H)

Production private prompt assembly bypass count on certified paths: **0** (mechanical CE-Q15 + EBH-4 composition).  
Legacy string builders removed from hot-path modules under gate scan.

## Tool-loop path (§I)

`IterativeToolOutputBlock` → CE assembly scope `acp_step` / iterative assembler → `engine.assemble` — not raw `ToolResult.content` append (see `test_mem_xint4_single_context_composition`).

## Boundary matrix (§J)

| Boundary | Verdict | Evidence |
| -------- | ------- | -------- |
| CE / Memory | PASS | CE-Q4, mem_xint4 LTM once |
| CE / RAG | PASS | CE-Q5, staged chunks |
| CE / UCL | PASS | `resolve_ucl_context_plan` inside engine; UCL not second budget owner |
| CE / ContextView | PASS | no CE ownership of principal visibility in runtime scan |

## Provider / engine replaceability (§K)

CE-Q9 custom provider; CE-Q11 custom engine via `engine_preset=custom` + `load_context_engine`; registry pinning tests in P1.9 catalog refs.

## Tenant 16Q (§L)

**Verdict: PASS** (local CE contribution — not TENANT-X promotion)

Adversarial: CE-Q3 foreign tenant fragment; `test_scope_isolation_rejects_foreign_tenant`; assembly request carries explicit `tenant_id`; fail-closed on scope mismatch.

## Typing / reflection (§M)

CE-Q2 typed contracts; CE-Q14 maintenance scripts + forbidden tokens scan; semantic `handles.get("messages")` removed from engine core (gate assertion).

## CE-Q1..CE-Q15 (catalog)

| Q | Verdict | Primary evidence |
| --- | ------- | ---------------- |
| CE-Q1 | PASS | `test_ce_q1_*`, integration graph CONTEXT_ASSEMBLED |
| CE-Q2 | PASS | typed ABI gate |
| CE-Q3 | PASS | tenant isolation |
| CE-Q4..Q6 | PASS | catalog integration/unit refs |
| CE-Q7..Q8 | PASS | policy authority / dedup |
| CE-Q9..Q11 | PASS | plugin + vendor gate |
| CE-Q12..Q15 | PASS | LLMAdapter boundary, provenance, tier0 imports, prompt injection scan |

**Package:** `uv run pytest tests/qualification/ce_01/ -p no:xdist` → **15 passed** @ HEAD.

## Test runs (§N)

| Run | Scope | Result |
| --- | ----- | ------ |
| #1 | `tests/unit/context/` | **309 passed, 21 failed** — `ContextProviderContext.runtime is required` (legacy handles-only tests) |
| #2–#4 | nexus context + wiring + integration paths + ce_01 + EBH-4-R1 | **225 passed, 18 failed** — same runtime requirement + `test_context_wiring` harness attribute + plan integration regex |
| CE-01 package alone | `tests/qualification/ce_01/` | **15/15 passed** |

Logs: `.tmp/session/ce-01/`.

## Pyright (§O)

```text
uv run pyright intergrax/context intergrax/contracts/context_assembly.py intergrax/runtime/nexus/context intergrax/runtime/execution/application_environment_context_composition.py
```

**82 errors** on baseline HEAD (pre-existing; no new CE semantic typing gate run in this session). Treat as baseline — exit criterion “0 new semantic errors” requires diff vs recorded baseline (not established here).

## FRZ contribution (§Q)

**global FRZ PASS delta = 0**  
**new FRZ-TEN PASS delta = 0**

Scoped evidence only: FRZ-OWN-01..03, FRZ-BND-01/02/04/05, FRZ-CTR-01/04/05/06, FRZ-TYP-01..04/06, FRZ-PLG-01/02, FRZ-RPL-01..03, FRZ-TRC-05 (local), FRZ-REG-01/02/03/09, FRZ-TEN-01/02/05/07/10/11/12 where CE tests apply.

## Non-uniform surfaces (§32)

| Item | Classification |
| ---- | -------------- |
| ContextManager fallback | **DOES NOT BLOCK CE-01** — gated off certified host wiring |
| ContextOrchestrator codebase | **DOES NOT BLOCK** — bounded preset |
| TOKEN-CE-1B / TOKEN-CE-2 | **OUTSIDE FROZEN CE-01 SCOPE** — planned optimization |
| Durable compaction runtime | **CE-02** — not CE-01 |
| Optional OTel completeness | **FUTURE NON-FROZEN CAPABILITY** |

## Unresolved findings (§S)

| ID | Class | Summary |
| -- | ----- | ------- |
| CE-01-R1 | **IN-SCOPE BLOCKER** | `DefaultNexusContextEngine` requires explicit `ContextAssemblyRuntimeDependencies` (`context_engine.py:193`); numerous unit tests still pass legacy `handles`-only `ContextProviderContext` → **39 failures** across `tests/unit/context/` and `tests/unit/runtime/nexus/context/test_context_plan_integration.py`. Production hot paths already use `build_context_assembly_runtime_dependencies` / `build_graph_provider_context_bundle`. |
| HARNESS-WIRING-01 | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | `test_build_harness_host_runtime_wires_context_manager_from_environment` — `HarnessHostInternalComposition` missing `_orchestration_backend` @ HEAD |

**unclassified findings = 0**

## Enterprise matrix (§48) — summary

Semantic owner, Nexus locus-only, single composition owner, certified paths CE semantics, single budget authority, boundaries, typing, pluginability, tenant local PASS, preflight on engine path: **PASS on architecture/code audit**.

Regression protection for full CE unit suites: **FAIL** until CE-01-R1.

## CE-01-R1 — Typed assembly runtime boundary (2026-10-02)

**START_HEAD:** `2df207b8012514c20ccb8e0f6fc2ede38e0d501c`  
**R1-A:** `ContextProviderContext.runtime` was semantic `Any` in `intergrax/context/contracts.py`.  
**R1-B:** Unit/integration tests used legacy `handles` (`runtime_config`, `messages`, …) without explicit `runtime` for `DefaultNexusContextEngine.assemble`.

**After type graph:** `intergrax/context/assembly_runtime.py` (`ContextAssemblyRuntime` / `ContextEngineRuntimeConfig` Protocols) ← structural ← `ContextAssemblyRuntimeDependencies` (Nexus) ← `ContextProviderContext.runtime` ← `DefaultNexusContextEngine`. No Tier-0 import of Nexus implementation types.

**Helpers:** `testing_support/context_assembly_test_runtime.py` (`build_test_assembly_runtime`, `provider_context_for_engine_assembly`, `provider_context_from_legacy_style_handles`).

**Gates:** `check_ce_canonical_semantic_handles.py` extended — forbidden `runtime: Any|object|dict|Mapping`; production legacy bridge call scan = 0.

| Run | Command | Result |
| --- | ------- | ------ |
| CE core | `uv run pytest -p no:xdist tests/unit/context/` | **330 passed** |
| Nexus context | `tests/unit/runtime/nexus/context/` | **228 passed** |
| CE-Q + wiring + integration | `ce_01/`, `test_context_wiring.py`, `test_context_engine_paths.py` | **566 passed** (combined batch) |
| Pyright (R1 slice) | `contracts.py`, `assembly_runtime.py`, `assembly_runtime_deps.py`, test helper | **0 errors** |

**Harness wiring:** `test_context_wiring` uses `resolve_context_manager_from_environment` (no `_orchestration_backend`).

**IN-SCOPE BLOCKER (R1 closure):** **0**

## Recommendation (§T)

```text
CE-01-R1 = READY FOR AUDIT
CE-01 = READY FOR AUDIT
CE-02 = NEXT / NOT ENTERED
PLUG-01 = NOT ENTERED
HARNESS-FINAL = NOT ENTERED
```

Independent audit must re-run CE-Q catalog + full CE unit/nexus context suites green before CLOSED.
