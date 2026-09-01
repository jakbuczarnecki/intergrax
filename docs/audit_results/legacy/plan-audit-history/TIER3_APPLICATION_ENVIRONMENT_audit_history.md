> **Migrated (AUDIT-PROTOCOL-RESET-R2):** Historical plan-satellite audit register.
> **Original path:** docs\project\maintainers\plans\satellites\TIER3_APPLICATION_ENVIRONMENT_implementation_history.md
> **Original role:** Plan satellite - audit history + LC closeout
> **Canonical audit ownership:** docs/audit_results/ (this file is historical evidence only)

# TIER3_APPLICATION_ENVIRONMENT - audit history + LC closeout

**Parent hub:** [`TIER3_APPLICATION_ENVIRONMENT.md`](../TIER3_APPLICATION_ENVIRONMENT.md)

## Phase H-APP - Tier-3 Application Environment (full configurability)

**Status:** **Done** (2026-06-03) - **43** deliverables; memory bridge via Phase MEM **Done**; source audit: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.  
**Prerequisites:** Phases **V**, **P-Ext**, **W-ML**, **W-OPS**, §4.1 **Done**.  
**Goal:** Close every **Partial** / **Gap** topic from the harness application-layer audit - full Tier-3 configurability of agent workspaces via `ApplicationEnvironmentProfile` and unified wiring (IDEAL §17), **without** Band 3 product agents (K.1/K.2).
**Priority ladder:** **Band 2e** (§4.0) - default implementation queue after §6.1 maintenance.  
**Execution order:** [§6.2x](.#62x-phase-h-app-execution-order-band-2e--active).

**Delivery rule:** One `H-APP.*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Out of scope (audit §7.7 - not counted in 43):** integration marketplace UI, catalog hot-reload, skill-as-LangGraph-pack, **IDEAL L4 runtime adaptation** (scheduled in [Phase W-ADAPT](.#phase-w-adapt--adaptive-harness-intelligence-l4-runtime), Band 2y), new Tier-0 integration categories without §5.2.4 RFC, K.1/K.2 business agents.

```text
Wave H0 - Docs & hygiene (5 tasks)
Wave H1 - ApplicationEnvironmentProfile + unified wiring (8 tasks)
Wave H2 - Identity, policy DSL, execution modes, V-SEC app hooks (8 tasks)
Wave H3 - Orchestration factory: graph spec, shadow/sandbox, Nexus composition (6 tasks)
Wave H4 - Context/Memory/Reliability/Observability profiles (8 tasks)
Wave H5 - Migrate all Tier-3 hosts + scaffold (5 tasks)
Wave H6 - Operational L3 sign-off (3 tasks)
Total: 43
```

### H-APP - Traceability (audit section → task IDs)

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
| §6 | EnvironmentProfile recommendation | H-APP.1.1–H-APP.1.5 · **APP-EVOL-8** (§22.6 bundles) |
| §6 (follow-up) | Per-app migration checklist | H-APP.5.1–H-APP.5.3 |

### H-APP - Master deliverables register (all 43 tasks)

| ID | Wave | Deliverable | Status | Priority | Location / acceptance |
|----|------|-------------|--------|----------|------------------------|
| H-APP.0.1 | H0 | **Harness terminology glossary** - Harness vs Tier-1 Nexus vs Tier-3 Application vs Tier-2 Agent vs Product; map to IDEAL §0.2 chain | **Done** | Medium | `intergrax_runtime_architecture.md` §5.3 + `IDEAL_HARNESS_AI_ARCHITECTURE.md` §26 cross-link |
| H-APP.0.2 | H0 | **Author guide: environment vs agent** - what belongs in `applications` vs `agents`; forbidden patterns | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` or `guides/AGENT_CREATION_GUIDE.md` |
| H-APP.0.3 | H0 | Fix `poc_template_application/host/wiring.py` - `manifest.integration_profile` (no `getattr`) | **Done** | High | Typed access; gate test |
| H-APP.0.4 | H0 | **`check_agent_registry_bypass.py`** - CI fails if Tier-2 agents import integrations/tools directly | **Done** | High | `scripts` + `pytest -m gate` |
| H-APP.0.5 | H0 | **Conformance test** - `ApplicationManifest` + `ApplicationBuildContext` round-trip (lab/legal/poc) | **Done** | High | `tests/unit/applications/test_manifest_conformance.py` |
| H-APP.1.1 | H1 | **`ApplicationEnvironmentProfile`** Pydantic model aggregating Tool/Skill/Modality/Policy/LLM/Context/Memory/Reliability/Observability/Orchestration/Identity profiles + `ApplicationFeatures` | **Done** | **Critical** | `intergrax/applications/contracts/environment_profile.py` |
| H-APP.1.2 | H1 | Extend **`ApplicationManifest`** with optional `environment` + `environment_defaults()` for `lab` / `product` | **Done** | **Critical** | `applications/contracts/manifest.py` |
| H-APP.1.3 | H1 | **`LLMProfile` slot** on environment - default adapter unless agent factory overrides | **Done** | High | Field + validation; no Tier-3 business logic |
| H-APP.1.4 | H1 | **`wire_application_environment(ctx, profile)`** - single Tier-3 entry for catalogs, modality, policy, tool/skill registries | **Done** | **Critical** | `applications/_shared/environment_wiring.py` |
| H-APP.1.5 | H1 | **`materialize_runtime_config(request, harness_ctx, env)`** - environment → `RuntimeConfig` | **Done** | **Critical** | `applications/_shared/runtime_config_bridge.py` |
| H-APP.1.6 | H1 | **`resolve_llm_adapter(env, agent_override)`** - precedence: agent factory > environment > platform default | **Done** | High | Typed resolver; unit tests |
| H-APP.1.7 | H1 | **`EnvironmentSkillToolConsistencyCheck`** - fail/warn if contract tools/skills not subset of environment | **Done** | High | `applications/_shared/conformance.py` |
| H-APP.1.8 | H1 | Gate tests: lab manifest + full `ApplicationEnvironmentProfile` | **Done** | High | `tests/unit/applications/test_environment_profile.py` |
| H-APP.2.1 | H2 | **`IdentityProfile`** - API key, tenant_required, role_claims_header, service_identities | **Done** | High | Part of `ApplicationEnvironmentProfile` |
| H-APP.2.2 | H2 | **`wire_application_identity(app, profile)`** - harness auth from profile | **Done** | High | `applications/_shared/identity_wiring.py` |
| H-APP.2.3 | H2 | **`ApplicationScopePolicy`** Protocol + static implementation - roles/scopes → tool_id / agent_id | **Done** | Medium | `applications/contracts` or `runtime/identity` |
| H-APP.2.4 | H2 | **`PolicyRulesProfile`** - declarative YAML/JSON rules + typed handler registry (no eval/getattr) | **Done** | **Critical** | `runtime/policy/rules` + schema |
| H-APP.2.5 | H2 | **`ExecutionMode`** enum: STRICT \| BALANCED \| EXPLORATORY → RuntimePolicies defaults | **Done** | High | `applications/contracts/execution_mode.py` |
| H-APP.2.6 | H2 | **`wire_policy_bundle(env)`** merges rules + fragments + ExecutionMode | **Done** | High | Extend `policy_wiring.py` |
| H-APP.2.7 | H2 | **`ApplicationSecurityProfile`** - per-app V-SEC toggles (prompt/tool/retrieval/tenant) | **Done** | Medium | Bridge to `runtime/architecture` V-SEC |
| H-APP.2.8 | H2 | Lab reference: `policy/rules/harness_lab.yaml` | **Done** | Low | `applications/lab_application/policy` + test |
| H-APP.3.1 | H3 | **`OrchestrationProfile`** - planner/classifier kinds, retry, long_running, max_delegation_depth | **Done** | High | Typed fields on environment |
| H-APP.3.2 | H3 | **`ApplicationGraphSpec`** - declarative multi-agent topology validated against roster | **Done** | High | `applications/contracts/graph_spec.py` |
| H-APP.3.3 | H3 | **`build_nexus_loop_from_environment(registry, integrations, env)`** | **Done** | **Critical** | `applications/_shared/nexus_factory.py` |
| H-APP.3.4 | H3 | **`wire_shadow_workspace(env)`** - ShadowWorkspaceManager paths, quotas, retention | **Done** | High | `applications/_shared/shadow_wiring.py` |
| H-APP.3.5 | H3 | **`wire_sandbox_sessions(env)`** - SandboxSessionManager + conditional `sandbox.exec` | **Done** | High | `applications/_shared/sandbox_wiring.py` |
| H-APP.3.6 | H3 | Integration test: lab graph spec echo → mock chain + trace | **Done** | Medium | `tests/integration/applications/test_lab_graph_spec.py` |
| H-APP.4.1 | H4 | **`ContextProfile`** - assembly options, budget presets, RAG/web toggles | **Done** | High | Pydantic model |
| H-APP.4.2 | H4 | **`MemoryProfile`** - user/org/long-term flags, retention, scope boundaries | **Done** | High | Pydantic model |
| H-APP.4.3 | H4 | Wire context/memory into `materialize_runtime_config` | **Done** | High | Phase MEM **MEM-1.*** - `memory_runtime_bridge.py`, `memory_wiring.py` |
| H-APP.4.4 | H4 | **`wire_task_memory_from_profile(env)`** - unify task memory under environment | **Done** | Medium | `_shared/task_memory_wiring.py` |
| H-APP.4.5 | H4 | **`ReliabilityProfile`** - idempotency, circuit breaker, checkpoint, scheduler | **Done** | High | Pydantic model |
| H-APP.4.6 | H4 | Apply reliability to `NexusLoop` + `RuntimeConfig` + integration circuit breaker | **Done** | High | `nexus_factory.py` |
| H-APP.4.7 | H4 | Gate test: long-running + idempotency via environment only | **Done** | Medium | `tests/unit/applications/test_reliability_profile.py` |
| H-APP.4.8 | H4 | **`ObservabilityProfile`** - trace, OTEL, metrics plugins, optional product debug surface | **Done** | Medium | Product hosts read-only debug option |
| H-APP.5.1 | H5 | **`lab_application`** - `build_lab_environment_profile` + refactor wiring/factory to unified environment | **Done** | **Critical** | No regression; gate + smoke |
| H-APP.5.2 | H5 | **`legal_application`** + **`research_application`** - product environment defaults + domain fragments | **Done** | High | Legal modality + skill bundles preserved |
| H-APP.5.3 | H5 | **`poc_template_application`** + **`docker_verify_application`** - environment template | **Done** | High | Scaffold emits profile stub |
| H-APP.5.4 | H5 | **Migration checklist** - per-file before/after (see table below) | **Done** | Low | `HARNESS_APPLICATION_LAYER_AUDIT.md` §7.6 + this phase |
| H-APP.5.5 | H5 | **`intergrax scaffold new-application`** - `environment_profile.py`, `policy/rules`, wired manifest | **Done** | Medium | CLI parity with H-APP.1 |
| H-APP.6.1 | H6 | Record **2 release cycles** via `record_harness_release_cycle.py --verify-gate` | **Done** | **Critical** | `build/architecture_hardening/release_cycles.json` |
| H-APP.6.2 | H6 | CI job: `phase_w_ops_evidence.py --enforce` on release tags | **Done** | High | `.github/workflows` |
| H-APP.6.3 | H6 | Mark Operational L3 **Signed off** in audit §4 with dates | **Done** | Low | `HARNESS_APPLICATION_LAYER_AUDIT.md` after H-APP.6.1 |

### H-APP - Per-application migration checklist (H-APP.5.4)

| Application | Files to refactor | Must wire via environment |
|-------------|-------------------|---------------------------|
| `lab_application` | `host/wiring.py`, `host/factory.py`, `host/tool_wiring.py`, `host/integration_wiring.py` | Full lab profile + harness tools + modality + plugins |
| `legal_application` | `host/wiring.py`, `host/factory.py`, `host/tool_wiring.py` | Product profile + legal skill bundle + optional modality |
| `research_application` | `host/wiring.py`, `host/factory.py` | Product profile + research agents roster |
| `poc_template_application` | `host/wiring.py`, `host/factory.py` | Minimal product/lab selectable template |
| `docker_verify_application` | `host/factory.py` | CI-oriented slim profile |

### H-APP - Explicitly deferred (not in the 43-task register)

| Topic | Reason |
|-------|--------|
| Integration marketplace UI | Out of P-Ext / audit §3.8 scope |
| Catalog hot-reload | Out of P-Ext scope |
| LangGraph skill packs | Separate initiative |
| IDEAL L4 adaptive / policy learning (runtime) | [Phase W-ADAPT](.#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · Band **2y** · AHIA |
| New Tier-0 integration categories | Requires canon §5.2.4 RFC (H-APP.0.2 documents process) |
| K.1 / K.2 business agents | Band 3 frozen (§6.3) |

### H-APP - Paydown log

| Date | H-APP ID | Summary |
|------|----------|---------|
| - | - | *(append row per merged PR)* |

**Suggested PR order:** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

---

---

### Phase H - Interaction Surfaces (§18)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| H.1 | Outbound webhook delivery | **Done** | §18 | Pluggable delivery + formatters; HTTP opt-in |
| H.2 | `InteractionAdapter` protocol | **Done** | §18 | Inbound → normalized `Task` |
| H.3 | Slack inbound lab path | **Done** | §18 | Debug API intake + signature stub |
| H.4 | HITL notification templates | **Done** | §42.10 | Reusable template + `notify_hitl_pause`; Slack/Teams formatters |
| H.5 | Teams parity | **Done** | §18 | Activity parser + HMAC verifier + debug intake tests |
| H.6 | Organization Worker demo | **Done** | §38 | E2E lab: intake → HITL → notification → resume |

---

---

### Phase N - Application Environment & Deploy Scaffold (Tier-3)

**Canon:** §7.4.8–§7.4.10  
**Goal:** From agent POC to **docker-pushable** dedicated lab/product host in minutes - same ergonomics as `new-agent`, with isolated `.env.example`, manifest, and Docker.

**Prerequisite:** Phase L complete; Phase M.3 (`IntegrationProfile`) available.

**Delivery rule (this phase):** One step per iteration - implement → summarize → update docs → present next step (see **§6.1**).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| N.0 | Architecture & plan documented | **Done** | §7.4.8–§7.4.10 | This section + runtime canon (2026-05-30) |
| N.1 | `ApplicationManifest` + `AgentBinding` models | **Done** | §7.4.10 | `intergrax/applications/contracts/manifest.py` |
| N.2 | Manifest conformance harness + unit tests | **Done** | §7.4.10 | `intergrax/applications/_shared/wiring.py` |
| N.2.1 | Unified agent initialization (builders / factories / context) | **Done** | §7.4.10 | `ApplicationBuildContext`, `build_application_registry`; lab + legal migrated |
| N.2.2 | Strongly typed `AgentBinding.mount(AgentClass, factory=...)` | **Done** | §7.4.10 | `type[Agent]` + callable factory; `deserialize()` for scaffold strings only |
| N.3 | `python -m intergrax.scaffold new-application` (profile `lab`) | **Done** | §7.4.8 | `new_application.py`, `agent_catalog.py`, `cli.py`; lab templates + smoke |
| N.4 | Scaffold profile `product` (fastapi_core skeleton) | **Done** | §7.4.8 | `new_application_product.py`; FastAPI Core + auth stub + `/health`; `--agents` list |
| N.5 | Docker templates under `applications/<app>/docker` | **Done** | §7.4.8 | Dockerfile + `.dockerignore` + `docker-compose.yml` + `build-docker.sh` / `.bat`; monorepo-root context |
| N.6 | Reference app `poc_template_application` (committed example) | **Done** | §7.4.8 | `applications/poc_template_application`; README three-command quickstart; gate smoke |
| N.7 | Backfill `.env.example` on existing apps | **Done** | §7.4.8 | `lab_application`, `legal_application`, `research_application`, `poc_template_application` |
| N.8 | `guides/AGENT_CREATION_GUIDE.md` Step 4E (dedicated application) | **Done** | - | Step 4E + Appendix F cross-links; gate doc test |
| N.9 | Acceptance `test_scaffold_application` (gate) | **Done** | - | `test_scaffold_acceptance.py` - lab/product E2E, CLI profiles, docker scripts |
| N.10 | Optional `new-stack` (agent + application in one CLI) | **Done** | - | `intergrax/scaffold/new_stack.py`; gate test in `test_scaffold_acceptance.py` |

#### N - Step-by-step implementation sequence

Execute **strictly in order**; do not skip ahead without completing acceptance for the current step.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | N.1 | Add `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` (Pydantic) | Unit tests pass; no scaffold yet |
| 2 | N.2 | Add `applications/_shared/conformance.py` (or mirror integrations pattern) | Manifest load + minimal registry build test |
| 3 | N.3 | Implement `new_application.py` + `lab` profile templates | `uv run python -m intergrax.scaffold new-application test_lab --profile lab --agents echo` creates tree; smoke test green |
| 4 | N.3b | Wire `build_parser()` subcommand; post-create hints (uvicorn, pytest, docker) | CLI prints next commands; gate test added (N.9 partial) |
| 5 | N.5 | Add Docker/docker-compose + build scripts to scaffold | `applications/<app>/docker/build-docker.sh` (or `.bat`) builds image from repo root |
| 6 | N.6 | Commit `applications/poc_template_application` from scaffold | README three-command quickstart verified |
| 7 | N.7 | Add per-app `.env.example` to legal, research, lab | Vars match each `settings.py`; no secrets committed |
| 8 | N.4 | Add `product` profile to scaffold | **Done** - `test_scaffold_product_application.py`; FastAPI Core + `/health` |
| 9 | N.8 | Update agent guide Step 4E | **Done** - scaffold lab/product, Docker scripts, three-command quickstart |
| 10 | N.9 | Full acceptance + `pytest -m gate` | **Done** - runtime E2E + `test_scaffold_acceptance.py` |

**Scaffold CLI (target interface):**

```bash
python -m intergrax.scaffold new-application my_lab \
  --profile lab \
  --agents echo,my_agent \
  --port 8091 \
  --prefix /v1/my_lab
```

**Out of scope for Phase N:**

- Separate `pyproject.toml` per application (stay monorepo + `pythonpath`)
- Auto-discovery of agents in `lab_application` (keep explicit wiring; manifest is declarative, not magic)
- Runtime sandbox (Tier-1) changes - only document distinction (§7.4.9)

#### Tier-3 application layer - readiness (2026-05-30)

**Status: ready** to generate new applications via scaffold. Checklist: [`applications/TIER3_READINESS.md`](../../../../../applications/TIER3_READINESS.md).

| Track | ID | Status | Notes |
|-------|-----|--------|-------|
| Engine | N.1–N.2.2 | **Done** | manifest, `build_application_registry`, conformance |
| Scaffold | N.3–N.4, N.10 | **Done** | `lab` + `product` + `new-stack` |
| Deploy | N.5–N.7 | **Done** | Docker scripts, `BUILD_AND_DEPLOY`, `.env.example` |
| Docs + gate | N.8–N.9 | **Done** | Step 4E, `test_scaffold_acceptance`, legal/research/lab manifest tests |
| Hardening | A.1–A.2 | **Done** | `test_legal_manifest_wiring`, tool_wiring assertions on scaffold |
| Optional CI Docker | B.1 | **Done** | `tests/integration/applications/test_poc_template_docker_build.py` (not in gate) |
| Product maturity | - | **Reference** | `legal_application` chat routes - extend scaffold `product` manually |

**Verify:**

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest -m gate -q
```

---

## Phase H-APP-DOC - Application interaction & orchestration authoring (Band 2ar - docs)

**Status:** **Done** (2026-06-09) - architecture canon §23; cross-refs to ORCHESTRATION §55, REASONING §9.4, NEXUS_EXECUTION_FLOW §3.1  
**Prerequisites:** Phase H-APP **Done** · Phase ORCH-STRAT **Done** · Phase COG-DOC **Done**  
**Goal:** Close authoring gaps for flexible Tier-3 postures (daemon, reactive, background) and multi-agent configuration without runtime changes.

**ADR:** [`ADR-FLOW-004`](../adr/entries/2026-06-09/ADR-FLOW-004.md) for `trigger_capabilities` (H-APP-DOC.2 / ORCH-CONFIG.2 **Done**). Authoring-only items need no ADR.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-DOC.1 | **Architecture §23** - posture catalog, routing matrix, scenario recipes | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` §23 |
| H-APP-DOC.2 | **`ApplicationGraphSpec.trigger_capabilities`** - optional seed guard (code) | **Done** | Medium | ORCH-CONFIG.2 · ADR-FLOW-004 · `test_graph_spec_to_plan.py` |
| H-APP-DOC.3 | **`intergrax/applications/USAGE.md` §** - orchestration configuration (ORCH-CONFIG / §56.13) | **Done** | Medium | Posture presets + harness proof links |
| H-APP-DOC.4 | **Scaffold `new-application` product** - interaction intake + scheduler optional wire | **Done** | Low | `INCLUDE_INTERACTIONS` / `INCLUDE_SCHEDULER`; legal host reference |

**Explicitly out of scope:** Nexus runtime fork; new coordination patterns (ORCH-5); COG-3 classifier implementation (tracked under ORCH-CONFIG.1 / COG-3.*).

**Canonical platform cases:** [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §56 · implementation register [`plan/ORCHESTRATION.md`](../plan/ORCHESTRATION.md) Phase **ORCH-CONFIG**.

---

## Phase H-APP-WIRING - Tier-3 execution surface parity (Band 2aw - planned)

**Status:** **Done** (2026-06-09) - **6/6 Done** · CFG host parity closeout (2026-06-09)  
**Audit source:** [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §59 · [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §23.7–§23.8 · FLOW-GAP-17–20  
**Prerequisites:** Phase H-APP **Done** · ORCH-6 **Done** · FLOW-CTL **Done** · REL-ADV **Done**  
**Goal:** Close **docs ↔ code discrepancies** where platform capabilities exist in Tier-1 but product hosts expose only sync `/run` - without Nexus forks.

**Priority ladder:** **Band 2aw** - recommended harness band after §6.1 gate maintenance (before Band 3 §6.3).

| ID | Gap / T3-GAP | Deliverable | Status | Priority | Acceptance |
|----|--------------|-------------|--------|----------|------------|
| H-APP-WIRING.1 | T3-GAP-01, T3-GAP-02 | Scaffold `INCLUDE_TASK_CONTROL` → optional `mount_harness_task_routes` + `apply_reliability_task_defaults` in `new-application` / `new-stack` | **Done** | **Critical** | `task_control_wiring.py` · `test_harness_task_control_wiring.py` |
| H-APP-WIRING.2 | T3-GAP-03, T3-GAP-04 | Adopt scheduler + task control on legal + research + poc_template reference hosts | **Done** | High | `legal_application` / `research_application` / `poc_template_application` factories |
| H-APP-WIRING.3 | T3-GAP-05, FLOW-GAP-18 | Optional `QueuedNexusExecutionAdapter` via `queue_worker_wiring.py` + `INCLUDE_QUEUE_WORKER` | **Done** | High | Legal host; scaffold env flags |
| H-APP-WIRING.4 | FLOW-GAP-20, CFG-14 | LKW hybrid daemon - explicit deferral in `local_workspace_application/ARCHITECTURE.md` | **Done** | Medium | §6.3 product backlog unchanged |
| H-APP-WIRING.5 | T3-GAP-01–04 | Task control + enricher + scheduler on assistant + dispute_sim + LKW hosts | **Done** | High | `intergrax_assistant_application` / `dispute_sim_application` / `local_workspace_application` factories |
| H-APP-WIRING-DOC.1 | - | Sync architecture §23.7–§23.8 + ORCH §59.2 host matrix | **Done** | Low | This phase closeout |

**Explicitly out of scope:** Nexus runtime changes; K.1/K.2; new queue transport.

**Cross-plan:** H-APP-WIRING.1 ↔ ORCH-6.5 · FLOW-CTL.6 · REL-ADV.7.

---

## Phase H-APP-CON - Application Environment Architecture canon (APP-CON)

**Status:** **Done** (2026-06-11) - architecture §24–§51 frozen; APP-CON · APP-PROD master registers **Done** - see [Master backlog](.#master-implementation-backlog-app-unified)
**Prerequisites:** Phase H-APP **Done** · H-APP-DOC **Done** · H-APP-WIRING **Done**  
**Goal:** Deliver **symmetric authoring canon** to ACP for Tier-3 - contracts, facades, hooks, checklists - without a new domain pair or Nexus fork.

**ADR:** no ADR needed for documentation-only tranche; **ADR-APP-001** recommended when mounting `ApplicationHost` into production pipeline (APP-CON-1 **Done**).

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-CON-DOC.1 | Architecture §24–§51 + TOC + fidelity matrix (this plan) | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` + §Architecture fidelity matrix |
| H-APP-CON-DOC.2 | Hub § Application in harness environment | **Done** | High | `intergrax_runtime_architecture.md` |
| H-APP-CON-DOC.3 | Cross-ref ACP §39 → TIER3 §39 canonical home | **Done** | Low | ACP §39.8 pointer |
| APP-CON-1..8 | Host contracts - see [APP-CON master](.#app-con--host-contracts-architecture-25-32--42--48) | **Done** | **Critical** | middleware · env state · hooks · artifacts |
| APP-PROD-1..9 | Release gates - see [APP-PROD master](.#app-prod--release-gates-architecture-40--46) | **Done** | High | `check_application_production_gates.py` |
| APP-CON-DX.* | Author + audit DX | **Done** | Medium | `APPLICATION_CREATION_GUIDE.md` · `check_tier3_audit_prompt.py` |

**Explicitly out of scope:** `Application.on_next_orchestration_step()`; new domain pair; Nexus runtime changes for product-specific orchestration.

**Rejected (documented in architecture §28.2):** cloning ACP step loop at Tier-3.

---

## Phase H-APP-EVOL - Runtime evolution and governance (APP-EVOL)

**Status:** **Done** (2026-06-11) - architecture §49 documented; APP-EVOL-1..7 **Done**  
**Prerequisites:** H-APP-CON architecture **Done** · V-ALG.3 agent lifecycle **Done**  
**Goal:** Close operational gaps for large-scale Tier-3 - versioning, migration, capability sunset, agent certification, recovery contract, environment diff, application packaging - without Nexus or profile primitive changes.

**ADR:** no ADR needed for §49 documentation tranche; **ADR-APP-002** recommended when `EnvironmentSnapshot` becomes mandatory on STRICT intake (APP-EVOL-1).

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-EVOL-DOC.1 | Architecture §49 Runtime Evolution and Governance | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` |
| APP-EVOL-1 | `EnvironmentSnapshot` + intake `profile_snapshot_id` | **Done** | **Critical** | `test_environment_snapshot_wiring.py` |
| APP-EVOL-2 | `ApplicationMigration` schema + CI validator | **Done** | High | `check_application_migrations.py` |
| APP-EVOL-3 | `CapabilityAlias` + deprecation routing | **Done** | High | `test_capability_alias_wiring.py` |
| APP-EVOL-4 | `AgentCertification` + STRICT roster gate | **Done** | High | `test_agent_certification_gate.py` |
| APP-EVOL-5 | `ApplicationRecoveryContract` on profile | **Done** | High | `test_recovery_contract_wiring.py` |
| APP-EVOL-6 | `ApplicationEnvironmentDiff` + `doctor diff-app` | **Done** | Medium | `check_application_environment_diff.py` |
| APP-EVOL-7 | `ApplicationPackage` + dependency resolver | **Done** | Medium | `check_application_package.py` |
| APP-EVOL-8-DOC | Architecture §22.6 hierarchical bundles | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` §22.6 · ADR-APP-003 |
| APP-EVOL-8.1–8.5, 8.7 | Bundle models + shims + presets + shared pack | **Done** (M1–M2) | See [APP-EVOL-8 register](.#app-evol-8--hierarchical-profile-bundles-p1-arch-01) |
| APP-EVOL-8.6 | `spec_version` 2.0 nested canonical wire | **Done** (M3) | `ProfileMigration` extension · `with_spec_v2_wire()` · `apply_profile_migration()` |

**Explicitly out of scope:** marketplace UI; Nexus fork; Tier-3 cognition loop; second composition root.

---

## Phase H-APP-OPS - Platform operations canon (APP-OPS) - freeze tranche

**Status:** **Done** (2026-06-11) - architecture §50 documented; APP-OPS-1..4 **Done**  
**Prerequisites:** H-APP-EVOL §49 **Done** · V-CG.1–3 capability graph **Done**  
**Goal:** Close reference-platform gaps - capability graph at environment scope, application ownership, health scoring, application/environment registry - **without** changing frozen primitives (Nexus, ApplicationHost, profile, graph spec, envelope, hooks).

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-OPS-DOC.1 | Architecture §50 Platform Operations Canon | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` |
| H-APP-OPS-DOC.2 | §49.2.4 typed migrations (Profile/Graph/Envelope) | **Done** | High | sub-migration schemas in §49 |
| APP-OPS-1 | Env capability graph + blast radius STRICT gate | **Done** | **Critical** | `check_capability_graph_strict_deploy.py` |
| APP-OPS-2 | `ApplicationOperationalOwnership` + APP-PROD | **Done** | High | `check_application_ownership.py` |
| APP-OPS-3 | `EnvironmentHealthScore` + `doctor health-app` | **Done** | High | `check_application_health_score.py` |
| APP-OPS-4 | `ApplicationRegistry` + `EnvironmentRegistry` + CLI | **Done** | Medium | `check_application_registry.py` |
| APP-EVOL-2b | Typed migration validators | **Done** | High | `migration_wiring.py` per primitive |

**Freeze declaration:** Tier-3 **structural architecture** is complete at §51 for flat profile §22.1. **APP-EVOL-8** (§22.6 hierarchical bundles · P1-ARCH-01) - **M1–M3 Done** (2026-06-18); ADR-APP-003 (**accepted**). APP-* master backlog **Done**; layer completion audit (2026-06-14) = **Architecturally Mature** for reference hosts. Further work is P3/P4 backlog only.

---

## Phase H-APP-FREEZE - Cross-document governance consistency audit

**Status:** **Done** (2026-06-11)  
**Goal:** Verify semantic alignment between Tier-3, ACP, UAEP, IDEAL - no duplicate capability/registry/ownership/health definitions before architecture freeze.

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| H-APP-FREEZE-1 | `guides/GOVERNANCE_CONSISTENCY_AUDIT.md` | **Done** | Five audit questions answered |
| H-APP-FREEZE-2 | TIER3 §51 + ACP §19 cross-refs | **Done** | Canonical ownership matrix |
| H-APP-FREEZE-3 | §22 GovernanceProfile description fix | **Done** | Flags ≠ ownership |

**Outcome:** No structural conflicts. Glossary bans `CapabilityRegistry`. Architecture freeze **approved**. APP-EVOL-1..7 and APP-OPS-1..4 **Done** (2026-06-11); layer completion audit (2026-06-14) confirms doc ↔ code alignment.

---

## Tier-3 Layer Completion Audit (2026-06-14)

**Verdict:** **Architecturally Mature** - no P0/P1 architecture gaps; APP-* master backlog **Done**.

| ID | Finding | Priority | Resolution |
|----|---------|----------|------------|
| T3-LC-01 | Architecture §49–§50 section headers still labeled `(target)` while registers show **Done** | P2 | Headers synced to **Done** · status rows authoritative |
| T3-LC-02 | `GOVERNANCE_CONSISTENCY_AUDIT` listed `CapabilityAlias` as planned | P2 | Updated to **Done** APP-EVOL-3 |
| T3-LC-03 | `runtime_config_bridge` missing import for `derive_run_budget_from_context_policy` | P1 | LC-IMPL-1 - import from `context_runtime_bridge` |
| T3-LC-04 | `ApplicationGraphSpec.graph_version` / `OrganizationalPolicyEnvelope.envelope_version` not on models | P4 | Migration schema only; model fields deferred |
| T3-LC-05 | Ownership inherit manifest → profile not wired | P4 | Manifest gate sufficient; profile inherit deferred |
| T3-LC-06 | Queue worker scaffold-default (T3-GAP-05) | P3 | Opt-in by design · AUDIT-IDEAL-28.2 **Done** |
| T3-LC-07 | Marketplace UI + signed distribution channel | P4 | H-APP explicitly deferred |

### Tier-3 backlog (post-completion)

| ID | Priority | Item | Notes |
|----|----------|------|-------|
| T3-BL-P1-01 | P1 | Hierarchical profile bundles (`APP-EVOL-8` · P1-ARCH-01) | M1–M3 **Done** · ADR-APP-003 |
| T3-BL-P3-01 | P3 | Default `INCLUDE_QUEUE_WORKER` on product scaffold | Opt-in today; legal + scaffold only |
| T3-BL-P3-02 | P3 | `RunBudget` from `CostProfile` beyond context mirror | Partial COST-1; context bridge derives when unset |
| T3-BL-P4-01 | P4 | `graph_version` on `ApplicationGraphSpec` | Migration schema ready |
| T3-BL-P4-02 | P4 | `envelope_version` on `OrganizationalPolicyEnvelope` | Uses `schema_version` today |
| T3-BL-P4-03 | P4 | Profile inherit `ownership` from manifest | APP-OPS-2 manifest gate covers deploy |
| T3-BL-P4-04 | P4 | Integration marketplace UI | H-APP deferred · §6.3 |
| T3-BL-P4-05 | P4 | Signed `ApplicationPackage` registry channel | Local/git channel **Done**; marketplace channel future |

### Sprint register (2026-06-14 layer completion)

| Sprint | Scope | DoD |
|--------|-------|-----|
| LC-DOC | Doc sync §49–§50 headers · hub · governance audit · this register | No `(target)` on **Done** APP-* rows |
| LC-IMPL-1 | `runtime_config_bridge` import fix | `uv run pytest tests/unit/applications/ -q` green |

---

## Phase TIER3-LC - Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) - re-validates H-APP + APP-CON/PROD/EVOL/OPS; no open P0/P1  
**Goal:** Formal Full Harness LC closeout - gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| TIER3-LC-S1 | **Re-audit** - H-APP register + host verdict | **Done** | High | No P0/P1 |
| TIER3-LC-S2 | **Plan/architecture sync** - Full Harness LC note | **Done** | High | Domain pair consistent |
| TIER3-LC-S3 | **Gate verification** | **Done** | High | applications unit tests · host wiring gates |
| TIER3-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** CFG-14 LKW hybrid · queue worker scaffold-default · marketplace UI

### 6.1av Harness implementation queue - Tier-3 application environment audit maintenance (planned)

**Source:** Layer 22 audit (2026-06-18) - `TIER3_APPLICATION_ENVIRONMENT` · [`../audit_results/2026-06-18/TIER3_APPLICATION_ENVIRONMENT.md`](../audit_results/2026-06-18/TIER3_APPLICATION_ENVIRONMENT.md)  
**Priority ladder:** **Band 1** (§6.1) - product-deferred items + host hygiene; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **T3-MAINT-01** | Cross-ref | P2 | **Done** | CFG-14 LKW hybrid daemon - cross-ref [`ORCH-MAINT-02`](ORCHESTRATION.md#61av-harness-implementation-queue--orchestration-audit-maintenance-planned) + §6.3 product gate | Runbook cross-ref when product prioritizes |
| 2 | **T3-MAINT-02** | Docs | P3 | **Done** | Queue worker scaffold-default (T3-GAP-05) - opt-in documentation for hosts | APPLICATION_CREATION_GUIDE queue worker opt-in |
| 3 | **T3-MAINT-03** | Backlog | P4 | **Done** | Marketplace UI + signed distribution - explicit §6.3 defer register row | Plan §6.3 defer; no scope creep |
| 4 | **T3-MAINT-04** | Code | P4 | **Done** | T3-LC-04/05 - `graph_version` / ownership inherit on profile models (deferred schema) | Deferred schema documented; no migration until product need |

**Suggested PR order:** T3-MAINT-02 → T3-MAINT-01 → T3-MAINT-04 → T3-MAINT-03.

**Cross-domain:** ORCH-MAINT-02 · ORCH-MAINT-01 (queue worker).

---

*End of Tier-3 Application Environment Implementation Plan.*
