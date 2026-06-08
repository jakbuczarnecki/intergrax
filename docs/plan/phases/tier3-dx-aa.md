# Implementation Phases — Tier3 Dx Aa

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase H-APP — Tier-3 Application Environment (full configurability)

**Status:** **Done** (2026-06-03) — **43** deliverables; memory bridge via Phase MEM **Done**; source audit: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.  
**Prerequisites:** Phases **V**, **P-Ext**, **W-ML**, **W-OPS**, §4.1 **Done**.  
**Goal:** Close every **Partial** / **Gap** topic from the harness application-layer audit — full Tier-3 configurability of agent workspaces via `ApplicationEnvironmentProfile` and unified wiring (IDEAL §17), **without** Band 3 product agents (K.1/K.2).
**Priority ladder:** **Band 2e** (§4.0) — default implementation queue after §6.1 maintenance.  
**Execution order:** [§6.2x](#62x-phase-h-app-execution-order-band-2e--active).

**Delivery rule:** One `H-APP.*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Out of scope (audit §7.7 — not counted in 43):** integration marketplace UI, catalog hot-reload, skill-as-LangGraph-pack, **IDEAL L4 runtime adaptation** (scheduled in [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime), Band 2y), new Tier-0 integration categories without §5.2.4 RFC, K.1/K.2 business agents.

```text
Wave H0 — Docs & hygiene (5 tasks)
Wave H1 — ApplicationEnvironmentProfile + unified wiring (8 tasks)
Wave H2 — Identity, policy DSL, execution modes, V-SEC app hooks (8 tasks)
Wave H3 — Orchestration factory: graph spec, shadow/sandbox, Nexus composition (6 tasks)
Wave H4 — Context/Memory/Reliability/Observability profiles (8 tasks)
Wave H5 — Migrate all Tier-3 hosts + scaffold (5 tasks)
Wave H6 — Operational L3 sign-off (3 tasks)
Total: 43
```

### H-APP — Traceability (audit section → task IDs)

| Audit § | Topic | Task IDs |
|---------|--------|----------|
| §1 | Terminology harness vs application vs agent | H-APP.0.1–H-APP.0.2 |
| §2.3.2 | Identity ABAC/RBAC per application | H-APP.2.1–H-APP.2.3 |
| §2.3.3, §3.4 | Policy DSL, execution modes, V-SEC per app | H-APP.2.4–H-APP.2.8 |
| §2.3.4, §3.5 | Orchestration graph spec, Nexus factory | H-APP.3.1–H-APP.3.6 |
| §2.3.5, §3.6 | LLMProfile on application manifest | H-APP.1.3, H-APP.1.6 |
| §2.3.7, §3.6 | ContextProfile, MemoryProfile | H-APP.4.1–H-APP.4.4 |
| §2.3.8, §3.8 | ReliabilityProfile | H-APP.4.5–H-APP.4.7 |
| §3.1 | Typed composition, no getattr in hosts | H-APP.0.3, H-APP.5.4 |
| §3.3 | Skill/tool permission consistency | H-APP.1.7, H-APP.0.4 |
| §3.5 | Shadow workspace + sandbox wiring | H-APP.3.4–H-APP.3.5 |
| §3.7 | Product observability profile (optional debug) | H-APP.4.8 |
| §4 | Operational L3 release evidence | H-APP.6.1–H-APP.6.2 |
| §5 | Registry bypass prevention | H-APP.0.4 |
| §6 | EnvironmentProfile recommendation | H-APP.1.1–H-APP.1.5 |
| §6 (follow-up) | Per-app migration checklist | H-APP.5.1–H-APP.5.3 |

### H-APP — Master deliverables register (all 43 tasks)

| ID | Wave | Deliverable | Status | Priority | Location / acceptance |
|----|------|-------------|--------|----------|------------------------|
| H-APP.0.1 | H0 | **Harness terminology glossary** — Harness vs Tier-1 Nexus vs Tier-3 Application vs Tier-2 Agent vs Product; map to IDEAL §0.2 chain | **Done** | Medium | `intergrax_runtime_architecture.md` §5.3 + `IDEAL_HARNESS_AI_ARCHITECTURE.md` §26 cross-link |
| H-APP.0.2 | H0 | **Author guide: environment vs agent** — what belongs in `applications/` vs `agents/`; forbidden patterns | **Done** | Medium | `EXTENSION_AUTHOR_GUIDE.md` or `AGENT_CREATION_GUIDE.md` |
| H-APP.0.3 | H0 | Fix `poc_template_application/host/wiring.py` — `manifest.integration_profile` (no `getattr`) | **Done** | High | Typed access; gate test |
| H-APP.0.4 | H0 | **`check_agent_registry_bypass.py`** — CI fails if Tier-2 agents import integrations/tools directly | **Done** | High | `scripts/` + `pytest -m gate` |
| H-APP.0.5 | H0 | **Conformance test** — `ApplicationManifest` + `ApplicationBuildContext` round-trip (lab/legal/poc) | **Done** | High | `tests/unit/applications/test_manifest_conformance.py` |
| H-APP.1.1 | H1 | **`ApplicationEnvironmentProfile`** Pydantic model aggregating Tool/Skill/Modality/Policy/LLM/Context/Memory/Reliability/Observability/Orchestration/Identity profiles + `ApplicationFeatures` | **Done** | **Critical** | `intergrax/applications/contracts/environment_profile.py` |
| H-APP.1.2 | H1 | Extend **`ApplicationManifest`** with optional `environment` + `environment_defaults()` for `lab` / `product` | **Done** | **Critical** | `applications/contracts/manifest.py` |
| H-APP.1.3 | H1 | **`LLMProfile` slot** on environment — default adapter unless agent factory overrides | **Done** | High | Field + validation; no Tier-3 business logic |
| H-APP.1.4 | H1 | **`wire_application_environment(ctx, profile)`** — single Tier-3 entry for catalogs, modality, policy, tool/skill registries | **Done** | **Critical** | `applications/_shared/environment_wiring.py` |
| H-APP.1.5 | H1 | **`materialize_runtime_config(request, harness_ctx, env)`** — environment → `RuntimeConfig` | **Done** | **Critical** | `applications/_shared/runtime_config_bridge.py` |
| H-APP.1.6 | H1 | **`resolve_llm_adapter(env, agent_override)`** — precedence: agent factory > environment > platform default | **Done** | High | Typed resolver; unit tests |
| H-APP.1.7 | H1 | **`EnvironmentSkillToolConsistencyCheck`** — fail/warn if contract tools/skills not subset of environment | **Done** | High | `applications/_shared/conformance.py` |
| H-APP.1.8 | H1 | Gate tests: lab manifest + full `ApplicationEnvironmentProfile` | **Done** | High | `tests/unit/applications/test_environment_profile.py` |
| H-APP.2.1 | H2 | **`IdentityProfile`** — API key, tenant_required, role_claims_header, service_identities | **Done** | High | Part of `ApplicationEnvironmentProfile` |
| H-APP.2.2 | H2 | **`wire_application_identity(app, profile)`** — harness auth from profile | **Done** | High | `applications/_shared/identity_wiring.py` |
| H-APP.2.3 | H2 | **`ApplicationScopePolicy`** Protocol + static implementation — roles/scopes → tool_id / agent_id | **Done** | Medium | `applications/contracts/` or `runtime/identity/` |
| H-APP.2.4 | H2 | **`PolicyRulesProfile`** — declarative YAML/JSON rules + typed handler registry (no eval/getattr) | **Done** | **Critical** | `runtime/policy/rules/` + schema |
| H-APP.2.5 | H2 | **`ExecutionMode`** enum: STRICT \| BALANCED \| EXPLORATORY → RuntimePolicies defaults | **Done** | High | `applications/contracts/execution_mode.py` |
| H-APP.2.6 | H2 | **`wire_policy_bundle(env)`** merges rules + fragments + ExecutionMode | **Done** | High | Extend `policy_wiring.py` |
| H-APP.2.7 | H2 | **`ApplicationSecurityProfile`** — per-app V-SEC toggles (prompt/tool/retrieval/tenant) | **Done** | Medium | Bridge to `runtime/architecture` V-SEC |
| H-APP.2.8 | H2 | Lab reference: `policy/rules/harness_lab.yaml` | **Done** | Low | `applications/lab_application/policy/` + test |
| H-APP.3.1 | H3 | **`OrchestrationProfile`** — planner/classifier kinds, retry, long_running, max_delegation_depth | **Done** | High | Typed fields on environment |
| H-APP.3.2 | H3 | **`ApplicationGraphSpec`** — declarative multi-agent topology validated against roster | **Done** | High | `applications/contracts/graph_spec.py` |
| H-APP.3.3 | H3 | **`build_nexus_loop_from_environment(registry, integrations, env)`** | **Done** | **Critical** | `applications/_shared/nexus_factory.py` |
| H-APP.3.4 | H3 | **`wire_shadow_workspace(env)`** — ShadowWorkspaceManager paths, quotas, retention | **Done** | High | `applications/_shared/shadow_wiring.py` |
| H-APP.3.5 | H3 | **`wire_sandbox_sessions(env)`** — SandboxSessionManager + conditional `sandbox.exec` | **Done** | High | `applications/_shared/sandbox_wiring.py` |
| H-APP.3.6 | H3 | Integration test: lab graph spec echo → mock chain + trace | **Done** | Medium | `tests/integration/applications/test_lab_graph_spec.py` |
| H-APP.4.1 | H4 | **`ContextProfile`** — assembly options, budget presets, RAG/web toggles | **Done** | High | Pydantic model |
| H-APP.4.2 | H4 | **`MemoryProfile`** — user/org/long-term flags, retention, scope boundaries | **Done** | High | Pydantic model |
| H-APP.4.3 | H4 | Wire context/memory into `materialize_runtime_config` | **Done** | High | Phase MEM **MEM-1.*** — `memory_runtime_bridge.py`, `memory_wiring.py` |
| H-APP.4.4 | H4 | **`wire_task_memory_from_profile(env)`** — unify task memory under environment | **Done** | Medium | `_shared/task_memory_wiring.py` |
| H-APP.4.5 | H4 | **`ReliabilityProfile`** — idempotency, circuit breaker, checkpoint, scheduler | **Done** | High | Pydantic model |
| H-APP.4.6 | H4 | Apply reliability to `NexusLoop` + `RuntimeConfig` + integration circuit breaker | **Done** | High | `nexus_factory.py` |
| H-APP.4.7 | H4 | Gate test: long-running + idempotency via environment only | **Done** | Medium | `tests/unit/applications/test_reliability_profile.py` |
| H-APP.4.8 | H4 | **`ObservabilityProfile`** — trace, OTEL, metrics plugins, optional product debug surface | **Done** | Medium | Product hosts read-only debug option |
| H-APP.5.1 | H5 | **`lab_application`** — `build_lab_environment_profile` + refactor wiring/factory to unified environment | **Done** | **Critical** | No regression; gate + smoke |
| H-APP.5.2 | H5 | **`legal_application`** + **`research_application`** — product environment defaults + domain fragments | **Done** | High | Legal modality + skill bundles preserved |
| H-APP.5.3 | H5 | **`poc_template_application`** + **`docker_verify_application`** — environment template | **Done** | High | Scaffold emits profile stub |
| H-APP.5.4 | H5 | **Migration checklist** — per-file before/after (see table below) | **Done** | Low | `HARNESS_APPLICATION_LAYER_AUDIT.md` §7.6 + this phase |
| H-APP.5.5 | H5 | **`intergrax scaffold new-application`** — `environment_profile.py`, `policy/rules/`, wired manifest | **Done** | Medium | CLI parity with H-APP.1 |
| H-APP.6.1 | H6 | Record **2 release cycles** via `record_harness_release_cycle.py --verify-gate` | **Done** | **Critical** | `build/architecture_hardening/release_cycles.json` |
| H-APP.6.2 | H6 | CI job: `phase_w_ops_evidence.py --enforce` on release tags | **Done** | High | `.github/workflows/` |
| H-APP.6.3 | H6 | Mark Operational L3 **Signed off** in audit §4 with dates | **Done** | Low | `HARNESS_APPLICATION_LAYER_AUDIT.md` after H-APP.6.1 |

### H-APP — Per-application migration checklist (H-APP.5.4)

| Application | Files to refactor | Must wire via environment |
|-------------|-------------------|---------------------------|
| `lab_application` | `host/wiring.py`, `host/factory.py`, `host/tool_wiring.py`, `host/integration_wiring.py` | Full lab profile + harness tools + modality + plugins |
| `legal_application` | `host/wiring.py`, `host/factory.py`, `host/tool_wiring.py` | Product profile + legal skill bundle + optional modality |
| `research_application` | `host/wiring.py`, `host/factory.py` | Product profile + research agents roster |
| `poc_template_application` | `host/wiring.py`, `host/factory.py` | Minimal product/lab selectable template |
| `docker_verify_application` | `host/factory.py` | CI-oriented slim profile |

### H-APP — Explicitly deferred (not in the 43-task register)

| Topic | Reason |
|-------|--------|
| Integration marketplace UI | Out of P-Ext / audit §3.8 scope |
| Catalog hot-reload | Out of P-Ext scope |
| LangGraph skill packs | Separate initiative |
| IDEAL L4 adaptive / policy learning (runtime) | [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · Band **2y** · AHIA |
| New Tier-0 integration categories | Requires canon §5.2.4 RFC (H-APP.0.2 documents process) |
| K.1 / K.2 business agents | Band 3 frozen (§6.3) |

### H-APP — Paydown log

| Date | H-APP ID | Summary |
|------|----------|---------|
| — | — | *(append row per merged PR)* |

**Suggested PR order:** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

---

## Phase DX — Developer Authoring Experience (fast environment + agent builds)

**Status:** **Done** (2026-06-02) — **47/47** deliverables **Done** in master table; gate **533+ passed**.  
**Prerequisites:** Phase **H-APP** **Done** (typed `ApplicationEnvironmentProfile`, `wire_application_environment`, `build_harness_host_runtime`). Phases **N**, **P-Ext**, **S** scaffold baseline **Done**.  
**Goal:** Make building **Tier-3 application environments** and **Tier-2 agents** trivial for Python developers — LangGraph-like mental model (state/steps → graph → run), **measurable** time-to-first-run (TTFRun), progressive disclosure (minimal → standard → production), and **UI-ready** serialized specs for Phase 2 (non-developer environment builder).  
**Priority ladder:** **Band 2f** (§4.0) — **closed for core path**; residual IDs are **infrastructure** follow-ups, not Band 3.  
**Scope split:** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business).  
**Execution order:** [§6.2y](#62y-phase-dx-execution-order-band-2f--mostly-done).

**Delivery rule:** One `DX-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Strategic split:**

| Phase | Audience | Outcome |
|-------|----------|---------|
| **DX (this phase)** | Python developers | Import contracts → define typed agents → configure environment → run HTTP/MCP in minutes |
| **Phase 2 (future — not DX)** | Business users via UI | Visual builder over same Pydantic/YAML specs (`DX-7.*` prepares artifacts only) |

**Target metrics (enforced by DX-3.5, DX-8.1):**

| Metric | Baseline (2026-06-03) | Target after DX |
|--------|----------------------|-----------------|
| **TTFRun** (scaffold → successful `POST …/run`) | ~45–90 min (docs + wiring) | **≤15 min** guided; **≤60 s** CI smoke |
| Author-edited files (hello world) | ~12–25 | **≤4** (`--minimal`) |
| Author LOC (excluding generated boilerplate) | ~200–400 | **≤120** |
| Commands to first run | 3+ | **1** (`intergrax run`) |
| Scaffold H-APP alignment | Partial (`factory.py` legacy path) | **100%** |

**LangGraph mapping (author mental model — implement in DX-0.2):**

| LangGraph | Intergrax (DX target) |
|-----------|------------------------|
| `State` fields | `AgentContract` + `RuntimeExecutionContext.metadata` |
| Node function | `@step` / `run_step` on `IntergraxAgent` |
| Conditional edges | `decide_after_step` → `AgentDecision` |
| `StateGraph.compile()` | `AgentGraph.build()` → `ApplicationGraphSpec` |
| `app.invoke()` | `HarnessApplication.serve()` / `POST /v1/…/run` |

**Out of scope (not counted in 47):** Band 3 product agents (K.1/K.2); visual graph editor UI; integration marketplace; catalog hot-reload; renaming `applications/` → `harness/` (canon §5.3.0 — **Application** = Tier-3 instance, **Harness** = platform); LangGraph skill pack import.

```text
Wave DX0 — Docs & traceability (4 tasks)
Wave DX1 — Scaffold/H-APP alignment fix (6 tasks) — P0 before facades
Wave DX2 — Authoring facades: HarnessApplication, AgentGraph, IntergraxAgent (6 tasks)
Wave DX3 — Minimal path + CLI + TTFRun gates (6 tasks)
Wave DX4 — Integration presets & picker (4 tasks)
Wave DX5 — Host hooks, YAML, observability/logging DX (8 tasks)
Wave DX6 — Tier hygiene + external projects (5 tasks)
Wave DX7 — UI engine prep: JSON Schema, spec versioning, catalog feed (5 tasks)
Wave DX8 — DX metrics & CI guards (3 tasks)
Total: 47
```

### DX — Traceability (audit gap → task IDs)

| Audit ref | Topic | Task IDs |
|-----------|--------|----------|
| L1 | Scaffold generates legacy + H-APP wiring in parallel | DX-1.1–DX-1.2, DX-1.6, DX-8.3 |
| L2 | No minimal hello harness (1–3 files) | DX-3.1, DX-2.1–DX-2.3, DX-3.2 |
| L3 | No fluent graph API | DX-2.2, DX-7.3 |
| L4 | No `HarnessApplication` / single entry class | DX-2.1, DX-2.6, DX-5.1–DX-5.2 |
| L5 | Monorepo-only (`pythonpath`) | DX-6.3–DX-6.5 |
| L6 | Tier-2 agents import `applications/_shared` | DX-6.1–DX-6.2 |
| L7 | IntegrationProfile slot knowledge burden | DX-4.1–DX-4.3, DX-4.2 |
| L8 | No `intergrax run` / `intergrax doctor` / TTFRun metric | DX-3.2–DX-3.3, DX-3.5, DX-8.1–DX-8.2 |
| L9 | Documentation sprawl, no single 15-min path | DX-0.1–DX-0.4, DX-3.6 |
| L10 | No JSON Schema / stable spec for UI phase 2 | DX-7.1–DX-7.5 |
| H-APP.5.3 gap | `poc_template` / scaffold `factory.py` not on `build_nexus_loop_from_environment` | DX-1.1, DX-1.3 |
| §6 responsibility table | Agent vs environment concerns split | DX-0.3 |
| Progressive disclosure | minimal → standard → production | DX-0.4, DX-3.4 |
| Architecture audit rec. | Product observability preset, trace_id in logs, event catalog | DX-5.5–DX-5.7 |
| Architecture audit rec. | Policy handler plugins (extend without core PR) | DX-5.8 |
| Do not weaken tiers | Doctor/checks enforce boundaries | DX-0.3, DX-3.3, DX-6.2, DX-8.3 |

### DX — Master deliverables register (all 47 tasks)

| ID | Wave | Deliverable | Status | Priority | Location / acceptance |
|----|------|-------------|--------|----------|------------------------|
| DX-0.1 | DX0 | **Phase DX register** in this plan + doc model row (§Documentation model) | **Done** | Low | This section + §6.2y |
| DX-0.2 | DX0 | **LangGraph ↔ Intergrax mapping** table (state, nodes, edges, compile, invoke) | **Done** | High | `EXTENSION_AUTHOR_GUIDE.md` §0 or `AGENT_CREATION_GUIDE.md` §1 |
| DX-0.3 | DX0 | **Responsibility matrix** — what belongs in agent vs environment (single canonical table) | **Done** | High | `EXTENSION_AUTHOR_GUIDE.md` §0 + cross-link canon §5.3.0 |
| DX-0.4 | DX0 | **Progressive disclosure** doc — minimal (`--minimal`) → standard scaffold → production (`expand`, Docker, MCP) | **Done** | Medium | `AGENT_CREATION_GUIDE.md` Step 4E § E.0 + `applications/USAGE.md` |
| DX-1.1 | DX1 | **Scaffold `factory.py`** — build `NexusLoop` only via `build_nexus_loop_from_environment(registry, env, …)` + integration bundle from `wire_application_environment` | **Done** | **Critical** | `intergrax/scaffold/new_application.py`, `new_application_product.py` |
| DX-1.2 | DX1 | **Scaffold default output** — remove generated `integration_wiring.py` + `tool_wiring.py`; retain via `--full` flag only | **Done** | **Critical** | Scaffold CLI + README in generated app |
| DX-1.3 | DX1 | **Migrate `poc_template_application/host/factory.py`** to H-APP factory pattern (no parallel legacy wiring) | **Done** | High | Parity with `lab_application`; gate smoke |
| DX-1.4 | DX1 | **Audit + fix** `legal_application` / `research_application` factories — single env path, no duplicate tool/integration wiring | **Done** | High | Host smoke tests green |
| DX-1.5 | DX1 | **Scaffold manifest** — embed `environment: ApplicationEnvironmentProfile…` at generation (not only lazy `environment_profile.py` fallback) | **Done** | High | Generated `manifest.py` |
| DX-1.6 | DX1 | **Gate test** — scaffold output: `factory.py` must not import `host.tool_wiring` / `host.integration_wiring` unless `--full` | **Done** | High | `tests/unit/scaffold/test_scaffold_harness_alignment.py` |
| DX-2.1 | DX2 | **`HarnessApplication` facade** — `.agents()`, `.integrations()`, `.graph()`, `.mode()`, `.llm()`, `.hooks()`, `.build()`, `.serve()` | **Done** | **Critical** | `intergrax/harness/app.py` |
| DX-2.2 | DX2 | **`AgentGraph` fluent builder** — nodes, edges, default agent, `on_error(retry=…)` → `ApplicationGraphSpec` | **Done** | **Critical** | `intergrax/applications/contracts/graph_builder.py` |
| DX-2.3 | DX2 | **`IntergraxAgent` base** + **`@step` decorator** — generates UAEP `get_steps` / `run_step` wiring | **Done** | **Critical** | `intergrax/agents/authoring/` |
| DX-2.4 | DX2 | **Decision helpers** — `continue_to()`, `complete()`, `delegate_to()` wrapping `AgentDecision` | **Done** | Medium | `intergrax/agents/authoring/decisions.py` |
| DX-2.5 | DX2 | **Unit test** — minimal `HarnessApplication` + `EchoAgent`/`IntergraxAgent` runs offline (no network) | **Done** | High | `tests/unit/harness/test_harness_application_minimal.py` |
| DX-2.6 | DX2 | **Public package** `intergrax.harness` — stable imports documented in author guide | **Done** | High | `intergrax/harness/__init__.py` |
| DX-3.1 | DX3 | **`new-stack --minimal`** — ≤4 author-facing files + smoke test (no Docker/MCP by default) | **Done** | **Critical** | `intergrax/scaffold/new_stack.py`, `new_application.py` `--minimal` |
| DX-3.2 | DX3 | **`intergrax run <module>:app`** — load `.env`, uvicorn, print route + sample curl | **Done** | High | `intergrax/cli/run.py` + `scaffold/cli.py` |
| DX-3.3 | DX3 | **`intergrax doctor`** — tier import violations, manifest/env conformance, scaffold freshness hint, TTFRun estimate | **Done** | High | `intergrax/cli/doctor.py` |
| DX-3.4 | DX3 | **`intergrax scaffold expand`** — promote minimal app → standard (Docker, MCP, debug, BUILD_AND_DEPLOY) | **Done** | Medium | `intergrax/scaffold/expand_application.py` |
| DX-3.5 | DX3 | **Acceptance test** `test_minimal_stack_ttf_run` — scaffold minimal → pytest → HTTP run **≤60s** in CI | **Done** | High | `tests/acceptance/dx/test_minimal_stack_ttf_run.py` |
| DX-3.6 | DX3 | **15-minute quickstart** — single numbered path: `new-stack --minimal` → edit agent → `intergrax run` → curl | **Done** | High | `AGENT_CREATION_GUIDE.md` Step 4E § E.0 |
| DX-4.1 | DX4 | **`IntegrationProfile` presets** — `.lab_stack()`, `.legal_stack()`, `.data_stack()`, `.observability_stack()` (typed, documented slugs) | **Done** | High | `intergrax/integrations/registry/presets.py` |
| DX-4.2 | DX4 | **`intergrax integrations pick`** CLI — emit profile fragment (postgres, redis, s3, prometheus, …) for `environment_profile.py` | **Done** | Medium | `intergrax/cli/integrations_pick.py` |
| DX-4.3 | DX4 | **Preset catalog table** in `INTEGRATIONS.md` + `EXTENSION_AUTHOR_GUIDE.md` | **Done** | Medium | `INTEGRATIONS.md` § Named integration presets |
| DX-4.4 | DX4 | **Gate tests** — each preset resolves with in-memory/sqlite stubs (no network) | **Done** | High | `tests/unit/integrations/test_integration_presets.py` |
| DX-5.1 | DX5 | **`ApplicationHost` Protocol/base** — override methods for environment control (intake, agent selection, finalize, error) | **Done** | High | `intergrax/harness/application_host.py` |
| DX-5.2 | DX5 | **Map host overrides → `HookPoint`** + optional `RuntimeEventBus` subscribe API on `HarnessApplication` | **Done** | High | Bridge in `intergrax/harness/hooks.py` |
| DX-5.3 | DX5 | **`HarnessApplication.from_yaml(path)`** — load `ApplicationEnvironmentProfile` + roster from `env.yaml` | **Done** | Medium | `intergrax/harness/yaml_loader.py` |
| DX-5.4 | DX5 | **Optional `agents.yaml`** — declarative `AgentBinding` list validated against importable classes | **Done** | Low | Same loader; schema test |
| DX-5.5 | DX5 | **Product scaffold observability preset** — `ObservabilityProfile` template (trace + optional read-only debug) | **Done** | Medium | `new_application_product.py` `environment_profile.py` (`otel_enabled`, debug override) |
| DX-5.6 | DX5 | **Structured log correlation** — inject `trace_id` / `run_id` in FastAPI middleware (lab + product factories) | **Done** | Medium | `intergrax/applications/_shared/logging_middleware.py` |
| DX-5.7 | DX5 | **Runtime event catalog table** — `RuntimeEventType` → emit phase → ops filter hints in canon §42 | **Done** | Low | `intergrax_runtime_architecture.md` §42.1.5; `phase_coverage.EVENT_OPS_FILTER_HINTS` |
| DX-5.8 | DX5 | **Policy rule handler plugins** — entry point group `intergrax.policy_rules` (mirror P-Ext pattern) | **Done** | Medium | `runtime/policy/rules/` + author guide § |
| DX-6.1 | DX6 | **`intergrax.agents.defaults`** — `harness_production_mode`, lab runtime config helpers (no Tier-3 import from agents) | **Done** | High | `intergrax/agents/defaults.py`; Tier-3 re-export in `runtime_defaults.py` |
| DX-6.2 | DX6 | **Fix reference agents** — `echo`, `research` (and scaffold template) must not import `applications/_shared` | **Done** | High | `agents/echo/`, `agents/research/` + `check_agent_registry_bypass` |
| DX-6.3 | DX6 | **`intergrax init <project>`** — cookiecutter: external repo, `pip install intergrax`, minimal harness layout | **Done** | High | `intergrax/scaffold/external_project/` template |
| DX-6.4 | DX6 | **CI smoke** — generated external template project pytest (fixture repo) | **Done** | Medium | `tests/integration/dx/test_external_project_template.py` |
| DX-6.5 | DX6 | **`pyproject` optional extra `[harness-author]`** — documented minimal dependency set for external apps | **Done** | Low | `pyproject.toml` + README |
| DX-7.1 | DX7 | **JSON Schema export** for `ApplicationEnvironmentProfile`, `ApplicationManifest`, `ApplicationGraphSpec` | **Done** | High | `scripts/export_harness_spec_schemas.py` → `build/harness_specs/` (CI) |
| DX-7.2 | DX7 | **`spec_version` on environment profile** + migration note in plan | **Done** | Medium | `environment_profile.py` |
| DX-7.3 | DX7 | **YAML round-trip tests** — graph + environment serialize/deserialize without loss | **Done** | High | `tests/unit/harness/test_spec_roundtrip.py` |
| DX-7.4 | DX7 | **Capability catalog JSON feed** — integrations/tools/skills slugs + labels for future UI builder | **Done** | Medium | `scripts/export_capability_catalog_feed.py` (CI) |
| DX-7.5 | DX7 | **Phase 2 UI boundary doc** — UI engine consumes DX-7 artifacts only; no parallel spec | **Done** | Low | Plan §Phase DX — UI boundary (below) |
| DX-8.1 | DX8 | **`intergrax doctor --ci`** — fail on tier violations, scaffold misalignment, TTFRun regression | **Done** | High | `.github/workflows/unit-tests.yml` |
| DX-8.2 | DX8 | **DX metrics in paydown** — record TTFRun seconds, author file count per release cycle | **Done** | Low | `scripts/record_dx_metrics.py` → `build/architecture_hardening/dx_metrics.json` |
| DX-8.3 | DX8 | **`check_scaffold_harness_alignment.py`** — CI script (complements DX-1.6 gate) | **Done** | High | `scripts/` + §6.1 maintenance list |

### DX — Explicitly deferred (not in the 47-task register)

| Topic | Reason |
|-------|--------|
| Visual graph editor / drag-and-drop UI | Phase 2 product; DX-7 only exports schemas |
| Rename `applications/` directory | Canon decision: Application = Tier-3 deployable instance |
| K.1 / K.2 business agents | Band 3 frozen (§6.3) |
| Full LangGraph runtime compatibility | Different execution model; mapping doc only (DX-0.2) |
| `phase_w_ops_evidence --verify-gate` hardening | W-OPS maintenance, not DX |

### DX — Paydown log

| Date | DX ID | Summary |
|------|-------|---------|
| 2026-06-03 | DX-1.1–DX-8.3 (core) | HarnessApplication, scaffold H-APP alignment, CLI run/doctor, presets, `check_scaffold_harness_alignment`; gate **518** |
| 2026-06-02 | Plan sync | Master table synced to codebase; **17** IDs remain **Pending** — [residual backlog](#dx--residual-backlog-infrastructure) |
| 2026-06-02 | DX residual closeout | `--minimal` stack, `expand`, doctor CI, spec export + round-trip, TTFRun acceptance, `agents.defaults.harness_production_mode`, docs quickstart; gate **533** |
| 2026-06-02 | DX-5.7 | §42.1.5 event catalog + `EVENT_OPS_FILTER_HINTS`; Phase DX **47/47 Done** |

**Suggested PR order (residual):** None — Phase DX infrastructure **Done**.

**Phase 2 UI boundary (DX-7.5):** A future visual builder must consume only versioned artifacts from `build/harness_specs/*.json` and `build/capability_catalog_feed.json` — not parallel Pydantic copies. Host behavior stays `HarnessApplication` / Tier-3 factories; UI edits serialize to the same `ApplicationEnvironmentProfile` + `ApplicationManifest` + `ApplicationGraphSpec` models validated by DX-7.1/DX-7.3.

### DX — Residual backlog (infrastructure)

**Not Band 3.** Platform DX rows **Done** (2026-06-02), including DX-5.7 (§42.1.5). No open DX IDs — see [§6.1z](#61z-harness-implementation-queue-consolidated).

---

## Phase AA — Agents & Applications Conformance (scaffold, docs, deploy)

**Status:** **Mostly Done** (2026-06-02) — **platform/conformance Done** (tier hygiene, ARCHITECTURE matrix, deploy triad, legal **scaffold** reset); **domain steps Deferred** (AA-LEG.2.2+); gate **534 passed**.  
**Prerequisites:** Phase **H-APP** **Done**, Phase **DX** **Mostly Done** (scaffold generators, `build_harness_host_runtime`, CLI, presets).  
**Goal:** Bring every **Tier-2** agent under `agents/` and every **Tier-3** host under `applications/` to a **documented, scaffold-aligned** state — fast authoring, full environment control (handlers, observability, policy), and **repeatable deploy** (Docker + deploy doc + `pyproject.toml` dependency contract per application). **Domain UAEP implementation is Band 3** — see [§6.3](#63-end-of-plan--deferred-product-work-only).  
**Priority ladder:** **Band 2g** (§4.0) — **platform rows closed**; only [AA residual](#aa--residual-backlog-infrastructure) + §6.1 maintenance.  
**Scope split:** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business).  
**Execution order:** [§6.2z](#62z-phase-aa-execution-order-band-2g--mostly-done).

**Delivery rule:** One `AA-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green → discuss scope in session before coding.

**Policy decision (2026-06-03):** **`agents/legal/` hard reset** — remove pre-architecture implementation; regenerate from `intergrax.scaffold new-agent legal` and re-implement only against UAEP + H-APP rules. Legacy tests become **behavioral spec** input, not code to preserve. **`legal_application`** follows the same reset cadence after the agent baseline exists (product shell only).

**Inventory (in scope):**

| Tier | Slug | Role |
|------|------|------|
| Agent | `echo` | Harness reference agent |
| Agent | `lab` | Lab mock agents (not product agents) |
| Agent | `legal` | **Hard reset** → scaffold baseline |
| Agent | `organization_worker` | Long-running / HITL demo |
| Agent | `problem_radar` | K.1 prototype (frozen until Band 3) |
| Agent | `research` | Multi-agent research prototype |
| Agent | `signoff_probe` | Appendix A sign-off exercise |
| Application | `lab_application` | Universal lab / debug superset |
| Application | `legal_application` | Legal product host (reset with agent) |
| Application | `poc_template_application` | Canonical minimal Tier-3 shell |
| Application | `research_application` | Research product host |

**Per-application deploy triad (mandatory for every Tier-3 host in this phase):**

Each `applications/<app>/` MUST document and maintain:

| Piece | Path / generator | Acceptance |
|-------|------------------|------------|
| **Docker** | `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/build-docker.sh`, `docker/build-docker.bat`, `docker/.dockerignore` | Image builds locally; health path matches manifest `route_prefix` |
| **Deploy doc** | `BUILD_AND_DEPLOY.md` | Generated/updated via `intergrax.applications._shared.build_deploy_doc.render_build_deploy_doc` (scaffold) or manual parity with scaffold output |
| **`ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md`** | Per agent/app directory | Cross-linked doc pair; local task queue in `IMPLEMENTATION_PLAN.md`; scaffold emits both (`intergrax/scaffold/doc_templates.py`) |
| **`pyproject.toml` deps** | Root `[project]` + `[project.optional-dependencies]` | Section in `applications/<app>/ARCHITECTURE.md`: which **core** deps apply, which **extras** (`harness-author`, `langgraph-legacy`, `llm-*`, `dev-ci`, integration extras) the host requires; no undeclared imports |

Scaffold already emits Docker + `BUILD_AND_DEPLOY.md` for **new** apps (`new-application`, `new-stack`). Phase AA **backfills and verifies** this triad on all four existing applications.

**Audit verdict (2026-06-03):**

| Area | Verdict | Gap → AA IDs |
|------|---------|----------------|
| Tier-2 structure vs canon | **OK** | Tier-3 imports removed; `legal` scaffold baseline; CI `check_agents_no_tier3_imports.py` |
| Tier-3 H-APP factory | **OK** | `build_harness_host_runtime` on lab/poc/legal/research; manifest `environment=` |
| Scaffold completeness | **OK** | Typed `can_handle`, `--reference`, deploy triad regression test |
| Documentation | **OK** | ARCHITECTURE.md matrix, guides, TIER3_READINESS — AA-D0.* **Done** |
| LangGraph independence | **Done** | `langgraph` not in core deps; `check_langgraph_not_required.py` — AA-LG.1 **Done** |
| Legal module | Reset required | AA-LEG.* + AA-LEGAPP.* |

```text
Wave AA0  — Register, scaffold checklist, LangGraph (done), deploy triad standard (5)
Wave AA1  — Platform docs meta: README, guides, TIER3_READINESS (7)
Wave AA2  — Legal agent HARD RESET (12)
Wave AA3  — legal_application reset + deploy triad (8)
Wave AA4  — echo agent (5)
Wave AA5  — signoff_probe agent (3)
Wave AA6  — problem_radar agent (5)
Wave AA7  — organization_worker agent (5)
Wave AA8  — research agent (+ summary) (6)
Wave AA9  — lab mock agents doc (2)
Wave AA10 — lab_application host (7)
Wave AA11 — poc_template_application host (5)
Wave AA12 — research_application host (6)
Total: 83 (incl. AA-LG.1 counted in AA0)
```

### AA — Traceability (audit topic → task IDs)

| Audit ref | Topic | Task IDs |
|-----------|--------|----------|
| A1 | Tier-2/3 separation (`agents` must not import `applications`) | AA-S0.2, AA-ECHO.2, AA-PR.2, AA-D0.3 |
| A2 | Scaffold agent `getattr` in `can_handle` | AA-S0.3 |
| A3 | Scaffold app missing manifest `environment=` | AA-S0.4, AA-POC.2, AA-RESAPP.2 |
| A4 | `lab_application` legacy Nexus factory | AA-LABAPP.2 |
| A5 | Legal pre-UAEP monolith | **AA-LEG.1–AA-LEG.12** (hard reset) |
| A6 | Per-agent architecture MD | AA-ECHO.1, AA-PR.1, AA-ORG.1, AA-RES.1, AA-SIG.1, AA-LEG.3 |
| A7 | Per-application architecture MD + deploy triad | AA-LABAPP.1, AA-POC.1, AA-RESAPP.1, AA-LEGAPP.1, AA-APP.0.1–AA-APP.0.3 |
| A13 | Doc pair `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` (scaffold + gate) | AA-D0.6, `intergrax/scaffold/doc_templates.py` |
| A8 | Root README completeness vs canon | AA-D0.1 |
| A9 | `AGENT_CREATION_GUIDE` / `TIER3_READINESS` stale | AA-D0.2–AA-D0.4 |
| A10 | LangGraph not required | AA-LG.1 (**Done**) |
| A11 | Docker + deploy script + pyproject per app | AA-APP.0.1–AA-APP.0.3, AA-*APP.*.4–*.6 |
| A12 | Legal application custom serving vs scaffold | AA-LEGAPP.3–AA-LEGAPP.5 |

### AA — Master deliverables register (all tasks)

#### Wave AA0 — Platform & scaffold foundation

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-0.1 | **Phase AA register** in this plan + §6.2z + doc model row | **Done** | Low | This section |
| AA-0.2 | **Scaffold ↔ H-APP checklist** table (new-agent / new-application / new-stack outputs) | **Done** | High | This section §AA scaffold matrix (below) |
| AA-S0.1 | Audit script: tier-2 must not import `applications` (extend `check_agent_registry_bypass` or sibling) | **Done** | High | `scripts/` + CI §6.1 |
| AA-S0.2 | **`new-agent`**: remove `getattr` from generated `can_handle` — typed `TaskContext` | **Done** | High | `intergrax/scaffold/new_agent.py` |
| AA-S0.3 | **`new-agent`**: optional `--reference` template (`HarnessReferenceAgent`) vs pure `Agent` | **Done** | Medium | `intergrax/scaffold/new_agent.py` |
| AA-S0.4 | **`new-agent`**: scaffold `contract.py` includes `skill_ids` placeholder + link SKILLS.md | **Done** | Medium | `intergrax/scaffold/new_agent.py` |
| AA-S0.5 | **`new-application`**: manifest always embeds `environment=ApplicationEnvironmentProfile…` | **Done** | High | `intergrax/scaffold/new_application.py` |
| AA-S0.6 | Document **`--full`** vs default scaffold (integration/tool wiring) | **Done** | Medium | `applications/USAGE.md` |
| AA-LG.1 | **LangGraph optional** — not in core deps; `langgraph-legacy` extra; `check_langgraph_not_required.py` | **Done** | High | `pyproject.toml`, CI |
| AA-APP.0.1 | **Deploy triad standard** — Docker + `BUILD_AND_DEPLOY.md` + pyproject extras section (canonical template) | **Done** | High | `applications/USAGE.md` §Deploy triad |
| AA-APP.0.2 | **Gate**: each existing `applications/*_application/` has `docker/`, `BUILD_AND_DEPLOY.md`, ARCHITECTURE deploy section | **Done** | High | `tests/unit/applications/test_application_deploy_triad.py` (incl. `local_workspace_application`) |
| AA-APP.0.3 | **Scaffold verify**: `new-application` output includes deploy triad (regression) | **Done** | High | `tests/unit/scaffold/test_scaffold_deploy_triad.py` |
| AA-D0.6 | **Gate**: doc pair `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` on listed agents/apps; cross-links | **Done** | High | `tests/unit/applications/test_agent_app_doc_pair.py` |

**AA scaffold matrix (generator vs H-APP target):**

| Output | `new-agent` | `new-application` (default) | `new-application --full` |
|--------|-------------|----------------------------|---------------------------|
| UAEP `Agent` + `steps/pipeline.py` | Yes | — | — |
| `contract.py` / `capabilities.py` | Yes | — | — |
| `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` | Yes | Yes | Yes |
| `manifest.py` + `AgentBinding` | — | Yes | Yes |
| `host/environment_profile.py` | — | Yes | Yes |
| `host/factory.py` → `build_harness_host_runtime` | — | Yes | Yes |
| `host/integration_wiring.py` | — | No | Yes |
| `host/tool_wiring.py` | — | No | Yes |
| `host/policy/rules/` | — | Yes (`.gitkeep`) | Yes |
| `docker/*` + `BUILD_AND_DEPLOY.md` | — | Yes | Yes |
| MCP + smoke tests | — | Yes | Yes |

#### Wave AA1 — Documentation meta (canon alignment)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-D0.1 | **Root `README.md`** — `HarnessApplication`, `intergrax` CLI, `poc_template` as Tier-3 reference, LangGraph optional, agent vs app matrix | **Done** | High | `README.md` |
| AA-D0.2 | **`docs/README.md`** — Phase AA row, last updated | **Done** | Low | `docs/README.md` |
| AA-D0.3 | **`AGENT_CREATION_GUIDE.md`** — DX paths (`intergrax run`, `doctor`, minimal stack); no stale Nexus-only flow | **Done** | High | `docs/AGENT_CREATION_GUIDE.md` |
| AA-D0.4 | **`applications/TIER3_READINESS.md`** — `environment_profile`, `build_harness_host_runtime`; deploy triad; no mandatory `tool_wiring` for all apps | **Done** | High | `applications/TIER3_READINESS.md` |
| AA-D0.5 | **`applications/USAGE.md`** — deploy triad + pyproject extras per host | **Done** | High | `applications/USAGE.md` |
| AA-D0.6 | **`EXTENSION_AUTHOR_GUIDE.md`** — LangGraph analogy only (not required) — verify post AA-LG.1 | **Done** | Low | Already partially done |
| AA-D0.7 | **Conformance index** in plan — agent/app status columns (this register) | **Done** | Low | Appendix row or §AA paydown |

#### Wave AA2 — `agents/legal` HARD RESET (decision: scaffold baseline only)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LEG.0.1 | **Record hard-reset decision** in plan + remove “incremental migration” as default for legal | **Done** | Critical | This section |
| AA-LEG.0.2 | **Archive tag** `legal-legacy-pre-aa` on git (pointer for forensic diff) | **Done** | High | Tag on parent of `bbce1bd` (pre hard-reset) |
| AA-LEG.0.3 | **Extract behavioral spec** from legacy tests → `agents/legal/SPEC_FROM_LEGACY.md` (requirements only) | **Done** | High | Before delete |
| AA-LEG.1.1 | **Delete** legacy `agents/legal/` tree (pipeline, governance, custom loop, tracing dupes) | **Done** | **Critical** | PR after AA-LEG.0.3 |
| AA-LEG.1.2 | **`python -m intergrax.scaffold new-agent legal --capability legal.review`** (force clean tree) | **Done** | **Critical** | `agents/legal/` matches scaffold layout |
| AA-LEG.1.3 | **`agents/legal/ARCHITECTURE.md`** — target UAEP graph, skills, tools, config, observability hooks (design-only until steps exist) | **Done** | High | English canonical doc |
| AA-LEG.2.1 | **Register** `legal` skill bundle on contract (`skill_ids`) per SKILLS.md | **Done** | High | `contract.py` |
| AA-LEG.2.2 | **UAEP steps** — port minimal slice from spec (one step per PR) | **Deferred** | High | `steps/` |
| AA-LEG.2.3 | **Remove** custom `legal_execution_loop`, `legal_tool_runtime_bridge` patterns — use Nexus `RuntimeToolGateway` only | **Deferred** | High | No parallel runtime |
| AA-LEG.2.4 | **Agent tests** — smoke + one spec-backed test per ported step | **Deferred** | High | `agents/legal/tests/` |
| AA-LEG.2.5 | **Retire** `ROADMAP.md` / `IMPLEMENTATION_PLAN.md` / `HOST_README.md` under agent — merge into `ARCHITECTURE.md` | **Done** | Medium | Single agent doc |
| AA-LEG.3.1 | **Gate**: `legal` agent imports no `applications.*`; no `getattr` on contract | **Done** | High | CI scripts |

**Explicitly NOT in legal reset:** Live LLM E2E product proof (Band 3 — K.6 / B.15 / S-Ops.4).

#### Wave AA3 — `applications/legal_application` (reset with agent)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LEGAPP.1 | **`ARCHITECTURE.md`** — manifest, environment profile, factory, auth, observability DB paths, MCP | **Done** | High | `applications/legal_application/` |
| AA-LEGAPP.2 | **Manifest** — `environment=ApplicationEnvironmentProfile.product_defaults(…)` inline | **Done** | High | `manifest.py` |
| AA-LEGAPP.3 | **Factory/serving** — align to `poc_template` + product settings; remove redundant `runtime_bridge` if superseded by `UnifiedTaskRunner` | **Done** | High | `host/factory.py`, `serving/` |
| AA-LEGAPP.4 | **Deploy triad** — verify/update `docker/*`, `BUILD_AND_DEPLOY.md` | **Done** | High | See AA-APP.0.1 |
| AA-LEGAPP.5 | **`pyproject.toml` deps section** in ARCHITECTURE — `harness-author`, LLM extras, optional `langgraph-legacy` N/A | **Done** | High | ARCHITECTURE §Dependencies |
| AA-LEGAPP.6 | **Host smoke** — `legal_tests/` green on scaffolded agent only | **Deferred** | High | After AA-LEG.2.2 |
| AA-LEGAPP.7 | **`.env.example`** parity with scaffold product profile | **Done** | Medium | `.env.example` |
| AA-LEGAPP.8 | **Remove** duplicate legal test trees if consolidated | **Deferred** | Low | `legal_tests/` vs agent tests |

#### Wave AA4 — Agent `echo`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-ECHO.1 | **`agents/echo/ARCHITECTURE.md`** — reference role, capabilities, skills, lab registration | **Done** | High | English |
| AA-ECHO.2 | **Remove Tier-3 imports** — inject `LabHarnessContext` from `lab_application` factory only | **Done** | **Critical** | `agents/echo/echo_agent.py` |
| AA-ECHO.3 | Align with **`HarnessReferenceAgent`** pattern documented in canon | **Done** | Medium | Code + doc |
| AA-ECHO.4 | **Tests** — import agent module without `applications` on PYTHONPATH | **Done** | High | `tests/unit/agents/` |
| AA-ECHO.5 | **README** — pointer to ARCHITECTURE only | **Done** | Low | `agents/echo/README.md` |

#### Wave AA5 — Agent `signoff_probe`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-SIG.1 | **`ARCHITECTURE.md`** — Appendix A sign-off flow, capability `signoff.probe` | **Done** | Medium | `agents/signoff_probe/` |
| AA-SIG.2 | Verify **scaffold parity** when AA-S0.2 lands (regenerate diff empty except domain) | **Done** | Low | `tests/unit/scaffold/test_signoff_scaffold_parity.py` |
| AA-SIG.3 | **README** → ARCHITECTURE link | **Done** | Low | |

#### Wave AA6 — Agent `problem_radar`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-PR.1 | **`ARCHITECTURE.md`** — K.1 placeholder, I/O schema, policy | **Done** | Medium | Frozen until Band 3 |
| AA-PR.2 | **Remove Tier-3 imports** (same pattern as echo) | **Done** | High | `problem_radar_agent.py` |
| AA-PR.3 | **Notebook + tests** documented in ARCHITECTURE | **Done** | Low | |
| AA-PR.4 | **Status** in plan §6.3 — no feature work until K.1 reprioritized | **Done** | Low | |
| AA-PR.5 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA7 — Agent `organization_worker`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-ORG.1 | **`ARCHITECTURE.md`** — HITL, long-running, `org.vendor_report` | **Done** | Medium | |
| AA-ORG.2 | **Remove `testing_support` import** — fake LLM via test fixture injection | **Done** | High | `organization_worker_agent.py` |
| AA-ORG.3 | **Scaffold-align** — add `contract.py`, `capabilities.py`, `steps/` if missing | **Deferred** | Medium | |
| AA-ORG.4 | **Lab manifest flag** + integration test | **Deferred** | Medium | `lab_application/manifest.py` |
| AA-ORG.5 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA8 — Agent `research` (+ `summary_agent`)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-RES.1 | **`agents/research/ARCHITECTURE.md`** — graph intent `research.pipeline`, two agents | **Done** | High | |
| AA-RES.2 | **Remove Tier-3 imports** from agents if any | **Done** | High | |
| AA-RES.3 | **`HarnessReferenceAgent`** alignment for Research/Summary | **Done** | Medium | |
| AA-RES.4 | **Skill ids** on contracts | **Deferred** | Medium | |
| AA-RES.5 | **Tests** — UAEP + graph delegation | **Deferred** | High | |
| AA-RES.6 | **README** merge into ARCHITECTURE | **Done** | Low | |

#### Wave AA9 — Agent `lab` (mocks)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LABAG.1 | **`agents/lab/README.md`** — mock agents purpose, not product Tier-2 | **Done** | Low | `agents/lab/README.md` |
| AA-LABAG.2 | **(Optional)** move mocks to `testing_support/` if they are test-only | **Won't fix** | Low | Until leadership requests — mocks stay under `agents/lab/` |

#### Wave AA10 — Application `lab_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LABAPP.1 | **`ARCHITECTURE.md`** — debug API, interaction, scheduler, manifest flags | **Done** | High | |
| AA-LABAPP.2 | **Migrate factory** to `build_harness_host_runtime` (retain rich wiring via env profile) | **Done** | **Critical** | `host/factory.py` |
| AA-LABAPP.3 | **`environment` in manifest** or documented single profile builder | **Done** | High | `manifest.py` / `_shared` |
| AA-LABAPP.4 | **Deploy triad** — verify `docker/*`, `BUILD_AND_DEPLOY.md` | **Done** | High | |
| AA-LABAPP.5 | **`pyproject.toml` deps** section in ARCHITECTURE | **Done** | High | |
| AA-LABAPP.6 | **Smoke tests** after factory migration | **Done** | High | `lab_application_tests/host/test_lab_host_smoke.py` + `tests/acceptance/agent_os/test_lab_application.py` |
| AA-LABAPP.7 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA11 — Application `poc_template_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-POC.1 | **`ARCHITECTURE.md`** — canonical Tier-3 lab shell (reference for new apps) | **Done** | High | |
| AA-POC.2 | **Manifest `environment=`** explicit (not only factory fallback) | **Done** | High | `manifest.py` |
| AA-POC.3 | **Deploy triad** verification | **Done** | Medium | |
| AA-POC.4 | **`pyproject.toml` deps** section | **Done** | Medium | |
| AA-POC.5 | **Link from root README** as “start here for new application” | **Done** | Medium | AA-D0.1 |

#### Wave AA12 — Application `research_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-RESAPP.1 | **`ARCHITECTURE.md`** — multi-agent HTTP, env vars, graph | **Done** | High | |
| AA-RESAPP.2 | **Manifest `environment=`** + `host/environment_profile.py` parity with scaffold | **Done** | High | |
| AA-RESAPP.3 | **Remove dead flags** `RESEARCH_USE_LEGACY_*` if obsolete | **Done** | Medium | `host/settings.py` |
| AA-RESAPP.4 | **Deploy triad** verification | **Done** | High | |
| AA-RESAPP.5 | **`pyproject.toml` deps** section | **Done** | High | |
| AA-RESAPP.6 | **Smoke tests** + `test_research_manifest_wiring` green | **Deferred** | High | |

### AA — Conformance matrix (living status)

| Module | Scaffold-aligned | ARCHITECTURE.md | Deploy triad | pyproject doc | Tier hygiene |
|--------|------------------|-----------------|--------------|---------------|--------------|
| `agents/echo` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/lab` | N/A (mocks) | README only | N/A | N/A | OK |
| `agents/legal` | Yes (scaffold) | **Done** | N/A | N/A | **OK** |
| `agents/organization_worker` | Partial | **Done** | N/A | N/A | **OK** |
| `agents/problem_radar` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/research` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/signoff_probe` | Yes | **Done** | N/A | N/A | OK |
| `applications/lab_application` | Yes | **Done** | **OK** | **Done** | H-APP factory |
| `applications/legal_application` | Yes | **Done** | **OK** | **Done** | H-APP |
| `applications/poc_template_application` | Yes | **Done** | **OK** | **Done** | OK |
| `applications/research_application` | Yes | **Done** | **OK** | **Done** | H-APP |

### AA — Residual backlog (infrastructure)

**Platform AA rows closed (2026-06-02).** Open infrastructure work: [§6.1z](#61z-harness-implementation-queue-consolidated) **V-REM** (2026-06-05) + ongoing **§6.1** maintenance.

| ID | Deliverable | Priority | Notes |
|----|-------------|----------|-------|
| AA-LABAG.1 | `agents/lab/README.md` — mock agents, not product Tier-2 | Low | **Done** — `agents/lab/README.md` |
| AA-LABAG.2 | (Optional) move lab mocks to `testing_support/` | Low | **Won't fix** until leadership requests |
| AA-SIG.2 | Scaffold parity diff test for `signoff_probe` | Low | **Done** — `tests/unit/scaffold/test_signoff_scaffold_parity.py` |
| AA-LABAPP.6 | Lab host smoke after H-APP factory | High | **Done** — unit + acceptance coverage |
| AA-LEG.0.2 | Git tag `legal-legacy-pre-aa` | High | **Done** — annotated tag on pre-reset commit |

### AA — Explicitly deferred (business / domain — Band 3)

| Topic | Task IDs | Reason |
|-------|----------|--------|
| Legal UAEP domain steps | AA-LEG.2.2–2.4, AA-LEGAPP.6, AA-LEGAPP.8 | Business logic on scaffold — [§6.3a](#63a-business-backlog-register-consolidated) |
| Research domain | AA-RES.4, AA-RES.5, AA-RESAPP.6 | Skills + graph tests — product prototype |
| Organization worker full scaffold | AA-ORG.3, AA-ORG.4 | Demo agent + lab roster |
| Lab host extra smoke | AA-LABAPP.6 | **Done** (2026-06-02 sync) — not blocking |
| K.1 / K.2 | Phase K | Band 3 — problem_radar / vendor discovery |
| Legal live LLM E2E | K.6 / B.15 / S-Ops.4 | Band 3 — CI budget |
| New product Tier-3 beyond four hosts | §6.3 | Product decision |

### AA — Paydown log

| Date | AA ID | Summary |
|------|-------|---------|
| 2026-06-03 | AA-0.1, AA-LEG.0.1 | Phase AA registered; **legal hard reset** policy recorded |
| 2026-06-03 | AA-LG.1 | LangGraph removed from core deps; CI `check_langgraph_not_required.py` |
| 2026-06-03 | AA-S0.1–S0.2, AA-S0.5, AA-APP.0.1–0.3, AA-ECHO.2, AA-PR.*, AA-LABAPP.2, AA-POC.2, AA-RESAPP.2, AA-LEG.1–1.3 | Tier hygiene, lab harness runtime, legal hard reset, deploy triad gate; gate **521** |
| 2026-06-02 | AA-S0.3, AA-D0.*, AA-* ARCHITECTURE, AA-LABAPP.3, AA-RESAPP.3 | `--reference` scaffold, docs matrix, lab manifest environment, tier import tests; gate **526** |
| 2026-06-02 | Plan sync | §4.0a scope split, DX/AA residual backlogs, §6.3a business register, master tables synced |
| 2026-06-02 | AA sync | AA-LABAPP.6 **Done**; AA-LABAG.2 **Won't fix**; §6.1z implementation queue |
| 2026-06-02 | AA-LEG.0.2, OPS-L3.1 | Tag `legal-legacy-pre-aa`; operational L3 evidence verified |

**Suggested session order (platform — complete):**  
See [§6.1z](#61z-harness-implementation-queue-consolidated). **Do not schedule** AA-LEG.2.* / AA-RES.5 / AA-ORG.3–4 in harness cadence — use [§6.3a](#63a-business-backlog-register-consolidated) after product decision.

---

