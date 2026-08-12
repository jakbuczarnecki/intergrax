# Tools

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/TOOLS.md`](../maintainers/plans/TOOLS.md)
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Audit layers:** 11  
**Audit instruction:** [`audit/TOOLS.md`](../maintainers/audit/TOOLS.md)
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (TOOLS canon).

- **Implement / audit default:** ToolRuntime path + plugin model + policy invoke (hub through production posture). Selection / invocation: [`satellites/TOOLS_selection_and_plugins.md`](satellites/TOOLS_selection_and_plugins.md). RuntimeConfig fields: [`satellites/TOOLS_runtime_config_reference.md`](satellites/TOOLS_runtime_config_reference.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/TOOLS.md`](../maintainers/plans/TOOLS.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/TOOLS.md`](../technical/guides/audit_slices/TOOLS.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/TOOLS_runtime_config_reference.md`](satellites/TOOLS_runtime_config_reference.md) | runtime config reference |
| [`satellites/TOOLS_selection_and_plugins.md`](satellites/TOOLS_selection_and_plugins.md) | selection and plugins |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Four-layer stack

```text
Tier-2  Agent (skill_ids, allowed_tools, ToolRequest)
        │
        ▼
Tier-0  Skill Library (MVP Done) — composable packs: tool_ids + prompts + policy — see [architecture/SKILLS.md](architecture/SKILLS.md)
        │
        ▼
Tier-0  Tool Library (rag.retrieve, jira.search_tasks, …)
        │
        ▼
Tier-0  Integration Library (IssueTracker, SearchProvider, VectorStore, …)
```

Skills are **not** tools — see architecture §7.1.8. Catalog: [architecture/SKILLS.md](architecture/SKILLS.md).

**Agents declare tool_ids.** **Applications enable tools** via `ToolProfile` and inject integrations via `ToolWiringContext`. **Integrations** remain vendor-swappable without agent changes.

---

## How wiring works (Phase O.2)

```text
Tier-3 application (tool_wiring.py)
        │
        ├── IntegrationProfile.resolve()  ──►  ToolWiringContext.from_integration_profile()
        │
        ▼
ToolProfile(enabled=[...], enabled_bundles=[...])
        │
        ▼
bootstrap_catalogs()  ──►  register_default_tools()  ──►  build_registry_from_profile(profile, ctx)
        │
        ▼
ToolRegistry  ──►  RuntimeToolInvoker  ──►  Agent / CatalogToolPlanner / MCP
```

**Example — enable tools from catalog profile:**

```python
from intergrax.tools.registry import (
    ToolProfile,
    ToolWiringContext,
    build_registry_from_profile,
    register_default_tools,
)
from intergrax.integrations import IntegrationProfile, register_default_integrations

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(issue_tracker="jira")
ctx = ToolWiringContext.from_integration_profile(profile)

registry = build_registry_from_profile(
    ToolProfile(enabled_bundles=["jira"]),
    ctx=ctx,
)
```

---

## Tool engine (implemented today)

Runtime tool engine (Phase O **Done** · **T-EXPAND Done** · **T14–T17 Done** — full **200-tool** catalog registered):

| Component | Path | Status |
|-----------|------|--------|
| `ToolContract` | `intergrax/tools/core/contracts.py` | **Done** — `ToolRiskLevel`, `ToolRetryPolicy`, metadata; invoker enforces timeout/retry |
| `ToolRegistry` | `intergrax/tools/registry/runtime.py` | **Done** |
| `ToolHandler` / `ToolExecutor` | `intergrax/tools/tool_executor.py` | **Done** |
| `ToolExecutionRequest` / `ToolExecutionResult` | `intergrax/tools/execution_models.py` | **Done** |
| `ToolProvider` protocol | `intergrax/tools/core/provider.py` | **Done** — accepts optional `ToolWiringContext` |
| `ToolCatalog` / `ToolProfile` / `ToolWiringContext` | `intergrax/tools/registry` | **Done** — Phase O.2; typed integration slots + `TaskMemoryViewBinding` / `shadow_workspace` (T-EXPAND) |
| `runtime_bound_catalog` | `intergrax/runtime/nexus/tools/runtime_bound_catalog.py` | **Done** — UAEP dispatch for `workspace.*` / `memory.*` / `harness.*` (incl. compare/export) · §42.12 |
| `register_default_tools()` / `build_registry_from_profile()` | `intergrax/tools/registry/bootstrap.py`, `factory.py` | **Done** |
| `RuntimeToolInvoker` | `intergrax/runtime/nexus/tools/invoker.py` | **Done** — validation, trace, error mapping |
| `RuntimeToolGateway` | `intergrax/runtime/nexus/tools/tool_gateway.py` | **Done** — capability aliases + registered catalog `tool_id` via `catalog_dispatch` (TOOL-ENG-2) |
| `catalog_dispatch` | `intergrax/runtime/nexus/tools/catalog_dispatch.py` | **Done** — per-id plan dispatch + gateway invoke (TOOL-ENG-1/2) |
| `BoundToolGateway` | `intergrax/runtime/nexus/tools/uaep_tool_gateway.py` | **Done** — UAEP §42.12 facade: `sandbox.exec` + 18 runtime-bound ids; catalog `tool_id`s delegate to `RuntimeToolGateway` (ADR-TOOL-001 · TOOL-ENG-2) |
| `CatalogToolPlanner` (LLM planner) | `intergrax/runtime/nexus/tools/catalog_tool_planner.py` | **Done** — OpenAI schema from registry via `ToolPlanningService` ([§Multi-tool execution](.#multi-tool-execution-semantics)) |
| `ToolPlanningService` | `intergrax/runtime/nexus/tools/tool_planning_service.py` | **Done** — native `generate_with_tools` or JSON fallback; `allowed_tool_ids` filter (TOOL-ENG-4) |
| `tool_planner_input` | `intergrax/runtime/nexus/tools/tool_planner_input.py` | **Done** — `tools_context_scope` assembly (TOOL-ENG-11) |
| `tool_selection` | `intergrax/runtime/nexus/tools/tool_selection.py` | **Done** — `ToolSelectionStrategy` router (TOOL-ENG-5/26/31/32) |
| `tool_loop` | `intergrax/runtime/nexus/tools/tool_loop.py` | **Done** — delegates to `ToolInvocationPattern` (TOOL-ENG-6,22) |
| `plan_context_invocation` | `intergrax/runtime/nexus/tools/plan_context_invocation.py` | **Done** — RAG/websearch/tools context for `ToolRuntime` (replaces retired pipeline steps) |
| `ToolInvocationPattern` | `intergrax/runtime/nexus/tools/tool_invocation_pattern.py` | **Done** — protocol + `pattern_for_mode()` (TOOL-ENG-16,21) · ADR-TOOL-003 |
| `SinglePassPattern` / `BoundedReactPattern` / `ParallelBatchPattern` | `intergrax/runtime/nexus/tools/patterns` | **Done** — shipped orchestration (TOOL-ENG-17,18,9) |
| `ToolInvocationAggregate` | `intergrax/runtime/nexus/tools/tool_invocation_aggregate.py` | **Done** — batch merge (TOOL-ENG-29) |
| `IdempotentToolInvoker` | `intergrax/runtime/tools/idempotent_invoker.py` | **Done** — exactly-once for `side_effects` + `idempotency_key` |
| `catalog_context` | `intergrax/runtime/nexus/tools/catalog_context.py` | **Done** — `rag.retrieve` / `websearch.query` dispatch via `plan_context_invocation` |
| `ToolAccessPolicy` | `intergrax/runtime/nexus/tools/tool_access_policy.py` | **Done** — plan-level filter (`ToolInvocationPlan`); modality intersect |
| `StaticToolScopePolicy` | `intergrax/runtime/tools/scope_policy.py` | **Done** — wired via `config.tool_scope_policy` in `RuntimeContext.build()` (TOOL-ENG-3) |
| `resolve_allowed_tools_from_config` | `intergrax/runtime/policy/tool_policy_resolution.py` | **Done** — merges `RuntimePolicyBundle.tool_access` into `ToolRuntime` / gateway |
| Legacy `ToolBase` | `intergrax/tools/tools_base.py` | **Deprecated** — use `ToolContract` (Phase O.7 Done) |

**Naming:** docs use **Tool engine** for the Tier-1 runtime stack below; **`ToolRuntime`** is the enforcement facade agents and Nexus MUST call (§42.12). Catalog types live in Tier-0 `intergrax/tools`.

---

## Tool execution pipeline

The **tool engine** is the Tier-1 stack that **selects** which catalog tools may run, **invokes** them through a single policy-checked path, and **logs** every attempt. Agents and graph nodes never call handlers or integrations directly.

**Read order:** this section (manifest) → [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §15–§17 (runtime sequence) → [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.12 (contracts).

```mermaid
flowchart TD
    subgraph Select["1 — Selection"]
        TP[ToolProfile bootstrap → ToolRegistry]
        SK[SkillResolver → AgentContract.allowed_tools]
        PB[RuntimePolicyBundle.tool_access]
        CTP[CatalogToolPlanner / EnginePlan tool_ids]
        LLM[LLM adapter tool_calls or text plan]
        TAP[ToolAccessPolicy.apply]
    end

    subgraph Orchestrate["2a — Invocation orchestration"]
        RBL[run_bounded_tool_loop / ctx.invoke_tool]
        TIP[ToolInvocationPattern — Done TOOL-ENG-16]
    end

    subgraph Invoke["2b — Atomic invoke"]
        TR[ToolRuntime.invoke / invoke_request]
        GW[RuntimeToolGateway / BoundToolGateway]
        RTI[RuntimeToolInvoker]
        IID[IdempotentToolInvoker optional]
        EX[ToolExecutor → ToolHandler]
        BE[Integration / RAG / sandbox backend]
    end

    subgraph Log["3 — Logging & governance"]
        TE[Nexus trace_event TraceComponent.TOOLS]
        EVT[RuntimeEventBus TOOL_REQUESTED / TOOL_*]
        MW[Middleware BEFORE/AFTER_TOOL_CALL]
        TRW[RunTraceWriter · tool trace payloads]
    end

    TP --> TR
    SK --> CTP
    PB --> TAP
    CTP --> LLM --> TAP --> RBL
    RBL --> TIP
    TIP --> TR
    TR --> GW --> RTI --> IID --> EX --> BE
    RTI --> TE --> EVT
    RTI --> MW
    TE --> TRW
```

### Phase responsibilities

| Phase | Question answered | Primary components | Tier |
|-------|-------------------|-------------------|------|
| **1 — Selection** | Which tools exist and which may this run use? | `ToolProfile`, `SkillResolver`, `resolve_allowed_tools_from_config`, `ToolSelectionStrategy`, `CatalogToolPlanner`, `ToolPlanningService`, `ToolAccessPolicy` | Tier-3 bootstrap + Tier-1 |
| **2a — Orchestration** | How is a **plan batch** executed (single / parallel / chain / ReAct)? | `ToolInvocationPattern` **Done** (TOOL-ENG-16) via `run_bounded_tool_loop` / `resolve_invocation_pattern()` | Tier-1 |
| **2b — Atomic invoke** | How is **one** tool call executed safely? | `ToolRuntime`, `RuntimeToolGateway`, `RuntimeToolInvoker`, `ToolExecutor`, `runtime_bound_catalog` | Tier-1 |
| **3 — Logging** | What happened, for audit and debug? | `trace_event`, `RuntimeEvent` (`TOOL_*`), security middleware, `RunTraceWriter`, agent/tool trace metadata — **must** use spine ([`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine); no private tool trace stores) | Tier-1 + observability |

### Entry paths — convergence on invoker

| Path | When used | Dispatch module | Reaches `RuntimeToolInvoker`? |
|------|-----------|-----------------|-------------------------------|
| **ACP `ctx.invoke_tool`** | Agent `on_next_step` / cognitive patterns (ReAct, etc.) | `BoundToolGateway` → `RuntimeToolGateway` | **Yes** — per `tool_id` on allow-list |
| **ToolRuntime catalog context** | `enable_rag` / `enable_websearch` or explicit `tool_ids` | `plan_context_invocation` + `catalog_context` | **Yes** — `rag.retrieve` / `websearch.query` |
| **Bounded tool loop** | ReAct / pattern with `max_tool_iterations > 1` | `tool_loop.run_bounded_tool_loop` | **Yes** — native tool-call rounds |
| **Capability `ToolRuntime.invoke`** | Legacy capability aliases (`use_rag`, `use_tools`) | `ToolRuntime` plan dispatch | **Partial** — prefer explicit `tool_ids` |
| **Engine plan tool_ids** | Nexus `EngineBackedNexusPlanner` node metadata | `ToolRuntime` via graph/agent host | **Yes** — when host wires planner output |
| **Tests / internal** | Unit tests, provider conformance | Direct `RuntimeToolInvoker.invoke` | **Yes** |

All successful catalog executions converge on **`RuntimeToolInvoker`** (optionally wrapped by **`IdempotentToolInvoker`**) — registry lookup, input/output schema validation, optional `ToolScopePolicy`, timeout/retry, error mapping, trace start/end.

Multi-call batches route through **`run_bounded_tool_loop`** / **`ctx.invoke_tool`**, which resolve and delegate to a configured **`ToolInvocationPattern`** before `RuntimeToolInvoker` (see [§Invocation patterns](.#tool-invocation-patterns-production-orchestration)).

### Selection detail (layers)

| Layer | Mechanism | What it filters | Applied when |
|-------|-----------|-----------------|--------------|
| **L0 Host catalog** | `ToolProfile` + `build_registry_from_profile()` | Which tools exist in runtime `ToolRegistry` | `RuntimeContext.build()` |
| **L1 Agent contract** | `AgentContract.allowed_tools` | Declared agent capability | Graph / UAEP bind |
| **L2 Skill packs** | `SkillResolver` → `tool_ids` on contract | Composed allow-list | Agent registration |
| **L3 Policy bundle** | `RuntimePolicyBundle.tool_access` (`StaticToolScopePolicy`) | Tier-3 static scope | `resolve_allowed_tools_from_config` |
| **L4 Modality** | `ModalityProfile` → `filter_tool_ids_by_modality_profile` | Media/ML plane tools | `ToolAccessPolicy.apply_modality_profile` |
| **L5 Plan filter** | `ToolAccessPolicy.apply` on `ToolInvocationPlan` | `use_rag` / `use_websearch` / `tool_ids` / `use_tools` | `ToolRuntime.invoke` |
| **L6 Schema narrowing** | `ToolSelectionStrategy` → `resolve_planner_allowed_tool_ids` | Subset passed to `ToolPlanningService` / `to_openai_tools` (see [§Production strategies](.#tool-selection-modes-production-strategies)) | `run_bounded_tool_loop` / `ctx.invoke_tool` (TOOL-ENG-5) |
| **L6b LLM planner** | `ToolPlanningService` → `generate_with_tools` | Model picks `tool_calls` from narrowed schema | `CatalogToolPlanner` |
| **L7 Invoker scope** | `ToolScopePolicy.is_allowed` on `RuntimeToolInvoker` | Per-call deny | **Done** — `scope_policy` from `RuntimeConfig` (TOOL-ENG-3) |

See [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) — cognition Plane 3 (Tool): `ToolPlanDecision` ≠ `AgentDecision` (§42.7).

### Invocation detail

```text
ToolExecutionRequest(run_id, step_id, tool_id, input, idempotency_key)
    → [optional] IdempotentToolInvoker (side_effects + idempotency_key)
    → RuntimeToolInvoker.invoke(state, agent_id, request)
        → ToolScopePolicy.is_allowed(agent_id, tool_id)   # when wired; else skipped
        → ToolRegistry.get(tool_id)
        → validate input_schema
        → ToolExecutor → ToolHandler → integration backend
        → validate output_schema (strict isinstance)
        → ToolRetryPolicy on contract (runtime-managed; agents MUST NOT retry)
    → ToolExecutionResult(success, output | error)
```

**Retry ownership:** tool retries are **R1 — ToolRuntime** layer only — [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md#retry-layers). Attempt metadata must be reconstructable via the observability spine ([`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine)).

`ToolRuntime.invoke_request(ToolRequest)` is the UAEP §42.12 surface; routes **sandbox**, **runtime-bound** ids, **capability aliases**, and **catalog `tool_id`s** via `BoundToolGateway` → `RuntimeToolGateway` (TOOL-ENG-2 **Done** · ADR-TOOL-001).

### Logging detail

| Signal | Mechanism | When |
|--------|-----------|------|
| Step trace | `state.trace_event(component=TOOLS, step=tool_invocation_*)` | Every invoker attempt (incl. denied scope) |
| Idempotency | `idempotency_cache_hit` trace step | Deduped side-effect replay |
| Runtime events | `TOOL_REQUESTED`, `TOOL_COMPLETED` / `TOOL_FAILED` / `TOOL_DENIED` | §42.12 |
| Ops filter | `ops:tool_audit` hint on tool events | [`OBSERVABILITY.md`](OBSERVABILITY.md) |
| Agent loop summary | `run_bounded_tool_loop` / `ctx.invoke_tool` → `state.tool_traces` (`ToolCallTrace`) | Single-pass planner |
| Budget | `enforce_tool_call_budget` → `BudgetEnforcer.check_tool_calls` | After each tool trace |
| Security | `MiddlewarePipeline` `BEFORE/AFTER_TOOL_CALL` | Guardrails / injection scan |
| Persisted run | `RunTraceWriter` / lab trace API | Post-mortem |

**Authoring:** [`AGENT_CREATION_GUIDE.md`](../technical/guides/AGENT_CREATION_GUIDE.md) Appendix J · **Audit:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11 · **Engine work:** [`plan/TOOLS.md`](../maintainers/plans/TOOLS.md) Phase **TOOL-ENG**.

---

## Platform tool plugin — developer path

**Task:** PLATFORM-PLUGIN-DOCS-3 · **Quickstart:** [`EXTENSION_AUTHOR_GUIDE.md`](../technical/guides/EXTENSION_AUTHOR_GUIDE.md) §3 · §16 · §17

| Delivery | Reference |
|----------|-----------|
| External wheel | `examples/platform_plugins/intergrax_reference_tool_plugin/` |
| Host-embedded | `examples/platform_plugins/local_embedded_tool_extension/` |
| In-repo minimal | `intergrax/tools/examples/custom_echo` |

**Sequence (both modes):** `ToolPlugin` → catalog registration → `ToolProfile` enablement → `ToolWiringContext` → `build_registry_from_profile` → `RuntimeToolInvoker`.

### Lifecycle

`ToolPlugin` registration is **catalog/bootstrap-time**. Handlers may own resources only when their domain design requires it. There is no generic Platform Plugin unload/shutdown manager. Integrations injected via `ToolWiringContext` follow host/domain lifecycle ownership.

### Failure behavior

| Condition | Behavior |
|-----------|----------|
| Duplicate bundle / tool id | `ValueError` from catalog or `ToolRegistry.register` |
| EP discovery/import failure | `PluginLoadError` |
| Bundle not on `ToolProfile` | Tool absent from runtime registry |
| Missing `ToolWiringContext` slot | Handler receives `None` — wire `IntegrationProfile` first |
| Qualification failure | Host `require_production_qualification` gate |
| Runtime invoke failure | `ToolExecutionResult` error / `TOOL_FAILED` trace event |

Bootstrap `on_conflict` policy: EXTENSION_AUTHOR_GUIDE §5.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Installed but tool missing | Enable discovery (`INTERGRAX_DISCOVER_PLUGINS`) |
| EP not discovered | Verify `intergrax.tools` entry-point group |
| Catalog row exists, not invokable | Add bundle/tool to `ToolProfile.enabled` |
| Qualification rejected | Host semantic evidence — not attestation |
| Handler dependency absent | Resolve `IntegrationProfile` → `ToolWiringContext` |
| Runtime invocation fails | Check schema, scope policy, integration backend |

Proof: `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py`

---

## Tool engine production posture (2026-06-10)

Full-stack audit of **Tier-0 catalog + Tier-1 tool engine** (selection → invoke → verify → log). Distinct from AUDIT-IDEAL-11.* (catalog sandbox/MCP/lint — **Done**).

### Maturity matrix

| Area | Posture | Notes |
|------|---------|-------|
| **Tier-0 catalog** (`ToolContract`, plugins, 200 tools) | **Production** | Contracts, exporters, provider tests, integration composition |
| **Single invoke** (`RuntimeToolInvoker`) | **Production** | Schema, timeout, retry, trace, idempotency wrapper |
| **Pipeline tool step** (`run_bounded_tool_loop` / `ctx.invoke_tool`) | **Done** | Planner wired; bounded loop via `tool_loop_step` (TOOL-ENG-6 · ADR-TOOL-002) |
| **Planner wiring** (`CatalogToolPlanner`) | **Done** | `wire_catalog_tool_planner_if_enabled` in `planner_bootstrap.py` (TOOL-ENG-0) |
| **Multi-tool / ReAct loop** | **Done** | `max_tool_iterations` + native `role=tool` chain (TOOL-ENG-6) |
| **Invocation pattern plugin** (`ToolInvocationPattern`) | **Production** | All shipped modes + `DeterministicChainPattern` (TOOL-ENG-16–24,28) |
| **Invoker test regression** (`modality_tool_trace`) | **Done** | TOOL-ENG-TEST.1 (S0) |
| **Deterministic tool chains** (output→input) | **Done** | `ToolChainSpec` + `DeterministicChainPattern` (TOOL-ENG-20) |
| **Parallel tool execution** | **Done** | `ParallelBatchPattern` + `max_parallel_tool_calls` (TOOL-ENG-9) |
| **Parallel semantic batch** | **Done** | `ParallelSemanticBatchPattern` (TOOL-ENG-25) |
| **Standard selection** (full schema → LLM) | **Production** | `FullCatalogSelectionStrategy` + `ToolPlanningService` (TOOL-ENG-0/4/5) |
| **Pre-filter selection** (keyword / skill / static) | **Production** | `ToolSelectionStrategy` — static, `skill_pack`, `retrieval_top_k` / `keyword_top_k` |
| **Semantic tool index** | **Done** | `ToolCatalogEmbedder` + `SEMANTIC` mode (TOOL-ENG-13) |
| **Hierarchical tool selection** | **Done** | Deterministic category→tool passes; optional LLM category pass opt-in (`tool_selection_hierarchical_llm_pass`, TOOL-MAINT-01b) |
| **`tool_ids` plan dispatch** | **Done** | `catalog_dispatch.invoke_catalog_tool_ids` (TOOL-ENG-1) |
| **§42.12 gateway** | **Done** | Catalog `tool_id` → invoker (TOOL-ENG-2); runtime-bound + sandbox unchanged |
| **`tool_scope_policy` wiring** | **Done** | `RuntimeToolInvoker` in `RuntimeContext.build()` (TOOL-ENG-3) |
| **Post-tool verification** | **Done** | `run_post_tool_verify` trace + enforce block (TOOL-ENG-7) — safety boundaries [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md#verification-safety-boundaries) |
| **Optional L1 critic on tool output** | **Planned** / **Deferred** (default **OFF**) | Post-invoke hook via CVL on high-risk tools only — not part of post-tool verification ship; see [Deferred runtime features](.#deferred-runtime-features-not-new-tools) |
| **AHI dynamic tool modes** | **Done** | `ToolEngineHook` + `recommend_tool_modes` (TOOL-ENG-10) |
| **Observability** | **Production** | Selection + pattern diag, budget ticks, `tool_traces` (TOOL-ENG-27/32) |

**Strategic focus (2026-06-12):** Phase **TOOL-ENG** **closed** — maintenance via gate scripts; deferred runtime features → [Phase TOOL-PRODUCT-ROI](.#phase-tool-product-roi--catalog-extension-by-product-value-planned).

---

## Phase TOOL-PRODUCT-ROI — Catalog extension by product value (Planned)

**Status:** Architecture & plan only — **not shipped**  
**Plan (1:1):** [`plan/TOOLS.md`](../maintainers/plans/TOOLS.md) — Phase TOOL-PRODUCT-ROI
**Policy:** One implementation ID per PR; register planned `tool_id`s only in matching task PRs.

**Purpose:** Extend the mature tool catalog (**200 shipped `tool_id`s**, **49** bundles, Full Harness LC **Done**) with **missing, high-ROI tools** for coding agents and change-audit agents — **not** general-purpose catalog padding. Existing families (RAG, filesystem, workspace, database, websearch, observability, eval, HITL, workflow, etc.) remain sufficient; add only gaps that improve **repository understanding** and **change safety**.

### Why TOOL-PRODUCT-ROI (product value, not catalog padding)

| Gap today | Harness need |
|-----------|--------------|
| Agents read files textually | Structured **code intelligence** — repo map, symbols, dependencies, architecture boundaries, diff risk |
| GitHub/GitLab context scattered | Read-only **Git / PR context** for audit agents before merge/approve tools |
| Unsafe direct writes | **Safe patch** preview + gated apply (phase 2 — write-capable) |
| Research claims without evidence chain | **Research evidence** layer above websearch/RAG (phase 3) |
| Full browser automation | Deferred — `browser.fetch_page` exists; interactive suite only if web-app agents become first-class |

### Wave 1 — Code Intelligence Tools (read-only, **P0**)

**Bundle id:** `code_intelligence` · **Public `tool_id` namespace:** `code.*` (bundle name and tool namespace are intentionally distinct).

All tools **read-only**, dispatch via **ToolRuntime**; backends may use `local_git` / workspace integrations (INT-P8.5) or in-process analyzers.

| `tool_id` | Purpose |
|-----------|---------|
| `code.repo_map` | Fast repository map: directories, modules, key files |
| `code.symbol_search` | Search classes, functions, methods, protocols, constants |
| `code.dependency_graph` | Module and layer dependency graph |
| `code.boundary_check` | Architecture boundary violations (e.g. tool bypassing ToolRuntime) |
| `code.diff_risk_analyze` | Pre-commit / pre-PR change risk assessment |
| `code.test_impact` | Tests to run after a change set |

**First-wave priority (highest ROI):** `code.repo_map`, `code.symbol_search`, `code.dependency_graph`, `code.boundary_check`, `code.diff_risk_analyze`.

### Wave 2 — Git / PR Context Tools (read-only, **P1**)

Read-only GitHub/GitLab (and local git) context for audit agents. **No** merge, approve, push, or apply-patch tools in this wave.

**Backend vs tools:** `local_git` (INT-P8.5) may expose approval-gated write backend operations (`apply_patch`, `commit`); Wave 2 `git.*` tools consume **read-only** operations only. Patch/commit surface ships later via `patch.*` tools and ToolRuntime policy gates.

| `tool_id` | Purpose |
|-----------|---------|
| `git.branch_diff` | Diff between branches |
| `git.pr_context` | PR metadata, description, review threads, changed files |
| `git.ci_status` | CI/check run status for branch or PR |

### Wave 3 — Safe Patch Tools (write-capable, **P2**)

Requires policy, idempotency, audit trail, optional HITL.

| `tool_id` | Purpose |
|-----------|---------|
| `patch.preview` | Show patch effect; validate allowed paths |
| `patch.apply_safe` | Apply patch only after preview + policy gate |

### Later families (product-gated)

| Family | Example `tool_id`s | Gate |
|--------|-------------------|------|
| **Browser automation** | `browser.navigate`, `browser.click`, `browser.fill_form`, `browser.screenshot`, `browser.extract`, `browser.network_requests`, `browser.console_messages` | Only if Intergrax hosts web-app agents as first-class |
| **Research evidence** | `research.evidence_pack`, `research.claim_verify`, `research.source_rank` | Research/audit agents needing claim↔source binding above websearch/RAG |

### Deferred runtime features (not new tools)

These extend existing engine paths; **default OFF**.

| Feature | Config / hook | Purpose |
|---------|---------------|---------|
| **Hierarchical LLM category pass** | `RuntimeConfig.tool_selection_hierarchical_llm_pass = false` | Optional LLM step when deterministic hierarchical selection picks wrong category on large catalogs; **must not** expand permissions or select outside policy allow-list (ADR-TOOL-005) |
| **Optional L1 critic on tool output** | Post-invoke hook on `RuntimeToolInvoker`: execution → output validation → optional L1 critic → allow / suspicious / block / require_hitl | High-risk tools only (e.g. `database.execute`, `filesystem.write_text`, `storage.delete`, `rag.purge_collection`, `platform.put_secret`, `collaboration.send_mail`, `patch.apply_safe`); see [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |

### Architectural boundaries (unchanged)

```text
Agent → Skill → ToolRuntime → Tool handler → Integration (optional)
```

- All new tools register as `ToolContract` + handler; agents **MUST NOT** call git parsers, LSP, or GitHub SDKs directly.
- Read-only waves **MUST NOT** perform writes or side effects (`side_effects=False`).
- Write-capable patch tools **MUST** use ToolRuntime policy, idempotency keys, trace spine, and HITL where configured.

### Explicit non-goals (TOOL-PRODUCT-ROI)

- Duplicating existing RAG, filesystem, workspace, websearch, or eval tools under new names
- Git write ops (merge, approve, push) before read-only context tools ship
- Global L1 critic on all read-only tools
- Browser automation suite without a Tier-3 product driver

---
