# Tier3 Application Environment — Implementation Plan

**Architecture (1:1):** [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**Cross-plan — Agent layer (ACP):** Tier-3 hosts supply `ApplicationEnvironmentProfile`, `AgentBinding`, intake `RequestIdentity`, and org envelope — consumed by agent `merge_environment` (architecture ACP §30 · TIER3 §39). Implementation synced in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 2** (`ACP-DX-2`, `ACP-DX-5`) and **Wave 6** (`ACP-ORG-1..2`). Host PRs that change profile merge order MUST update agent plan acceptance tests.

**Application authoring canon (APP-CON):** architecture §24–§51 — symmetric to ACP §12–§45 for Tier-3 environments. **Evolution canon (APP-EVOL):** architecture §49. **Operations canon (APP-OPS):** architecture §50. **Freeze audit:** [`guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](../guides/GOVERNANCE_CONSISTENCY_AUDIT.md). Phases **H-APP-CON** · **H-APP-EVOL** · **H-APP-OPS** · **H-APP-FREEZE** below.

**Fidelity rule:** Every architecture §20–§51 normative row MUST map to a plan ID in [§Architecture fidelity matrix](#architecture-fidelity-matrix--20-51) and a verification artifact in [§Fidelity verification gates](#fidelity-verification-gates). Completing the **open APP-\*** backlog is sufficient for implementation to match frozen architecture — no new primitives without ADR.

---

## Architecture fidelity matrix — §20–§51

Maps each architecture section to **plan phase**, **implementation status**, **code anchor**, and **acceptance test**. **Done** = architecture row is implemented and gated unless marked *doc-only*.

| Arch § | Topic | Plan IDs | Status | Code / test anchor |
|--------|-------|----------|--------|-------------------|
| §20 | Shadow workspace lifecycle | H-APP.3.4 · APP-CON-8 · APP-PROD-8 | **Done** | `shadow_wiring.py` · `workspace_cleanup_wiring.py` · `check_workspace_cleanup.py` |
| §21 | Sandbox lifecycle | H-APP.3.5 · APP-CON-8 · APP-PROD-8 | **Done** | `sandbox_wiring.py` · `workspace_cleanup_wiring.py` · `test_workspace_cleanup_wiring.py` |
| §22 | `ApplicationEnvironmentProfile` | H-APP.1.* | **Done** | `environment_profile.py` · `test_environment_profile.py` |
| §23 | Interaction postures | H-APP-DOC.* · H-APP-WIRING.* | **Done** | §23.7 matrix closed on reference hosts |
| §24 | `ApplicationManifest` / `AgentBinding` | N.1 · H-APP.1.2 | **Done** | `manifest.py` · `test_manifest_conformance.py` |
| §25 | `run_task` / `HarnessApplication` / `ApplicationHost` | APP-CON-1 · N.* | **Done** | `harness/app.py` · `test_application_host_wiring.py` |
| §26 | `ApplicationRunSummary` (Plane A) | ACP-OBS-2 · APP-CON-6 | **Done** | `application_run_summary_builder.py` · `run_artifact_bundle_builder.py` |
| §27 | Roster / registry assembly | N.2.1 · H-APP.1.4 | **Done** | `registry_assembly_resolver.py` |
| §28 | APP invariants (no app cognition loop) | H-APP-CON-DOC.* | **Done** | *doc-only* · rejected `on_next_orchestration_step` |
| §29–§31 | Terminology · control modes · facade | H-APP.0.* · APP-CON-DX.1 | **Partial** | DX appendix open |
| §32 | `ApplicationHost` hook surface | APP-CON-1 | **Done** | `application_host.py` · `hooks.py` |
| §32.6 | Hook ordering · conflicts · determinism | APP-CON-5 | **Done** | `hook_runtime_guard.py` · `middleware_hook_timeout_seconds` |
| §33 | Dual observability planes | ACP-OBS-* · H-APP.4.8 | **Done** | `test_application_run_summary_builder.py` |
| §34 | Per-agent `AgentBinding` / budget slice | H-APP.* · ACP §30 | **Done** | `merge_environment` · ACP plan Wave 2 |
| §35 | Use-case catalog UC-A* | APP-CON-7 | **Done** | `tier3_scenario_matrix_wiring.py` · `test_tier3_scenario_matrix.py` |
| §36 | Architecture synthesis | H-APP-CON-DOC.* | **Done** | *doc-only* |
| §37 | Pre-implementation APP-CON contracts | H-APP-CON-DOC.* | **Done** | *doc-only* |
| §38 | L4 execution stack | H-APP.3.3 · H-APP-WIRING | **Done** | `nexus_factory.py` · `build_harness_host_runtime` |
| §39 | `OrganizationalPolicyEnvelope` | ACP-ORG-* | **Done** | `org_policy.py` · `test_uc11_product_host_compliance.py` |
| §40 | APP-PROD gates | APP-PROD-1..9 | **Partial** | APP-PROD-7 · APP-PROD-9 **Done** · 6 · 8 open |
| §41 | Composition primitive separation | H-APP-CON-DOC.* | **Done** | *doc-only* |
| §42 | `ApplicationEnvironmentState` v2 | APP-CON-2 · APP-CON-3 | **Done** | `environment_state.py` · lifecycle middleware |
| §43 | Budget / token governance | ACP-TOK-* · APP-CON-3 · APP-PROD-7 | **Done** | see [Cross-plan §43](#cross-plan--43-budget--token-governance) |
| §44 | Scenario test matrix | APP-CON-7 | **Done** | `check_tier3_scenario_matrix.py` · `-m tier3_scenario` |
| §45 | New application checklist | APP-CON-DX.1 · N.* | **Partial** | scaffold + guide |
| §46 | Production readiness criteria | APP-PROD-* · ACP-PROD-* | **Partial** | §46 + agent gates |
| §47 | Developer mental model | APP-CON-DX.1 | **Partial** | *doc-only* in arch |
| §48 | Application artifacts | APP-CON-4 · APP-CON-6 | **Done** | `application_artifacts.py` · `run_artifact_bundle.v1` on summary |
| §49 | Runtime evolution | APP-EVOL-1..7 · APP-EVOL-2b | **Partial** | APP-EVOL-1/2/2b/3 **Done** · §49.8 register |
| §50 | Platform operations | APP-OPS-1..4 | **Partial** | APP-OPS-1 **Done** · `capability_graph_deploy_gate.py` |
| §51 | Cross-doc consistency | H-APP-FREEZE-* | **Done** | `GOVERNANCE_CONSISTENCY_AUDIT.md` |

---

## Master implementation backlog (APP-* unified)

Single register for all open architecture rows. **Execution order:** [§6.2y](#62y-phase-app-backlog-execution-order-post-freeze).

### APP-CON — host contracts (architecture §25–§32 · §42 · §48)

| ID | Arch § | Deliverable | Status | Acceptance |
|----|--------|-------------|--------|------------|
| APP-CON-1 | §25 · §32 | `ApplicationHost` in `build_harness_host_runtime` | **Done** | `test_application_host_wiring.py` |
| APP-CON-2 | §42 | `ApplicationEnvironmentState` v2 | **Done** | `test_environment_state_and_artifacts.py` |
| APP-CON-3 | §42 · §43 | Nexus lifecycle updates `app_env_state.v1` (phase, budget, HITL) | **Done** | `test_application_environment_state_lifecycle.py` |
| APP-CON-4 | §48 | Artifact ref models | **Done** | `application_artifacts.py` |
| APP-CON-5 | §32.6 | Hook timeout · error→BLOCK · audit events | **Done** | `test_hook_runtime_guard.py` · product 250ms timeout |
| APP-CON-6 | §26 · §48 | `RunArtifactBundle` on `ApplicationRunSummary.metadata` | **Done** | `test_task_finisher_artifact_bundle.py` |
| APP-CON-7 | §35 · §44 | Scenario matrix gate — UC-A* minimum per posture | **Done** | `tier3_scenario_matrix_wiring.py` · `check_tier3_scenario_matrix.py` · `-m tier3_scenario` |
| APP-CON-8 | §20–§21 | Shadow/sandbox refs in env state + lifespan cleanup | **Done** | `workspace_cleanup_wiring.py` · `test_workspace_cleanup_wiring.py` |
| APP-CON-DX.1 | §31 · §45 · §47 | Author guide APP appendix (mental model + checklist) | Planned | `AGENT_CREATION_GUIDE.md` or `APPLICATION_CREATION_GUIDE.md` |
| APP-CON-DX.2 | §37 | Regenerate domain audit prompt for §24–§51 | Planned | `generate_domain_audit_prompts.py` |

### APP-PROD — release gates (architecture §40 · §46)

| ID | Arch § | Deliverable | Status | Acceptance |
|----|--------|-------------|--------|------------|
| APP-PROD-1 | §40.2 | `check_application_production_gates.py` | **Done** | script exits 0 |
| APP-PROD-2 | §40.2 | Reference hosts use `build_harness_host_runtime` | **Done** | H-APP-WIRING |
| APP-PROD-3 | §40.2 | `ApplicationHost` mounted when provided | **Done** | `test_application_host_wiring.py` |
| APP-PROD-4 | §40.2 | Manifest conformance | **Done** | `test_manifest_conformance.py` |
| APP-PROD-5 | §40.2 | Deploy triad | **Done** | `test_application_deploy_triad.py` |
| APP-PROD-6 | §40.2 | `check_environment_state_usage` lint | Planned | CI script; hooks use typed state |
| APP-PROD-7 | §40.2 · §43 | `check_budget_enforcement` on STRICT product hosts | **Done** | `check_budget_enforcement.py` · product manifests `budget_slice` |
| APP-PROD-8 | §20–§21 | `check_workspace_cleanup` lifespan hooks | **Done** | `check_workspace_cleanup.py` · `test_check_workspace_cleanup.py` |
| APP-PROD-9 | §40.2 | Wire APP-PROD-1 into `pytest -m gate` / CI | **Done** | `test_check_application_production_gates.py` · CI `gate-governance-tier` |

### APP-EVOL — evolution (architecture §49)

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| APP-EVOL-1 | `EnvironmentSnapshot` + intake `profile_snapshot_id` | **Done** | `test_environment_snapshot_wiring.py` · ADR-APP-002 |
| APP-EVOL-2 | `ApplicationMigration` schema + CI validator | **Done** | `application_migration.py` · `check_application_migrations.py` |
| APP-EVOL-2b | `ProfileMigration` / `GraphSpecMigration` / `OrgEnvelopeMigration` | **Done** | `migration_wiring.py` typed validators per §49.2.4 |
| APP-EVOL-3 | `CapabilityAlias` + deprecation routing | **Done** | `capability_alias_wiring.py` · `check_capability_alias_registry.py` |
| APP-EVOL-4 | `AgentCertification` + STRICT roster gate | Planned | non-PRODUCTION blocked in product hosts |
| APP-EVOL-5 | `ApplicationRecoveryContract` on `ReliabilityProfile` | Planned | product ARCHITECTURE template + test |
| APP-EVOL-6 | `ApplicationEnvironmentDiff` + `doctor diff-app` | Planned | pre-deploy CI diff |
| APP-EVOL-7 | `ApplicationPackage` + dependency resolver | Planned | `new-stack` emits package manifest |

### APP-OPS — platform operations (architecture §50)

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| APP-OPS-1 | Env capability graph + blast radius STRICT gate | **Done** | `check_capability_graph_strict_deploy.py` · `test_capability_graph_deploy_gate.py` |
| APP-OPS-2 | `ApplicationOperationalOwnership` on manifest | **Done** | `check_application_ownership.py` · `test_operational_ownership_gate.py` |
| APP-OPS-3 | `EnvironmentHealthScore` + `doctor health-app` | Planned | release score artifact |
| APP-OPS-4 | `ApplicationRegistry` + `EnvironmentRegistry` + CLI | Planned | `intergrax apps list` / `envs list` |

---

## Cross-plan — §43 budget / token governance

Architecture §43 is **implemented jointly** with ACP §25.4–§25.5. Tier-3 configures; harness enforces; agents read.

| Arch §43 row | Owner plan | ID | Status |
|--------------|------------|-----|--------|
| `CostProfile` / `budget_reaction` config | TIER3 (this file) | H-APP.1.1 `CostProfile` | **Done** |
| `AgentBinding.budget_slice` | TIER3 + ACP | H-APP.1.2 · ACP §34 | **Done** |
| Token metering rollups | ACP | **ACP-TOK-1** | **Done** |
| Kernel hard cap + block LLM | ACP | **ACP-TOK-2** | **Done** |
| Host notify / HITL / `custom_hook` | ACP + TIER3 | **ACP-TOK-3** · APP-CON-3 | **Done** |
| CI gate | ACP | **ACP-TOK-CI** | **Done** |
| APP-PROD-7 host gate | TIER3 | **APP-PROD-7** | **Done** |

**Fidelity rule:** §43 **Done** — ACP-TOK-* complete, APP-CON-3 seeds `ActiveBudgetState`, APP-PROD-7 gates STRICT product manifests.

---

## Fidelity verification gates

Run after any Tier-3 PR touching hosts, contracts, or wiring:

```bash
# Tier-3 unit + host contracts
uv run pytest tests/unit/applications/ -q

# APP-PROD-1 (wire to gate via APP-PROD-9)
python scripts/check_application_production_gates.py

# Harness tier boundaries
python scripts/check_harness_no_getattr.py
python scripts/check_agent_registry_bypass.py

# Domain pair + journal
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py

# Full gate (includes agent + platform)
uv run pytest -m gate -q
```

**Architecture-complete Tier-3 DoD (target):** all rows in [Master backlog](#master-implementation-backlog-app-unified) **Done** · fidelity matrix all **Done** · `GOVERNANCE_CONSISTENCY_AUDIT.md` glossary respected · no §51 naming violations.

---

## §6.2y Phase APP backlog execution order (post-freeze)

Recommended PR sequence — one APP ID per PR:

```text
1.  APP-PROD-9      wire production gates to CI
2.  APP-CON-3       env state lifecycle sync on hooks — **Done**
3.  ACP-TOK-1..3    (agent plan) budget enforcement — unblocks §43
4.  APP-PROD-7      budget gate on STRICT hosts
5.  APP-CON-5       hook timeout / error handling
6.  APP-CON-6       artifact bundle on ApplicationRunSummary
7.  APP-CON-8       shadow/sandbox cleanup + APP-PROD-8 — **Done**
8.  APP-EVOL-1      EnvironmentSnapshot on intake — **Done**
9.  APP-OPS-1       capability graph STRICT deploy gate — **Done**
10. APP-OPS-2       application ownership on manifest — **Done**
11. APP-CON-7       scenario matrix tests — **Done**
12. APP-EVOL-2/2b   migrations — **Done**
13. APP-EVOL-3..7   evolution + packaging — **3 Done** · 4–7 open
14. APP-OPS-3/4     health score + registries
15. APP-CON-DX.*    author guide + audit prompt
```

**Cross-plan:** steps 3–4 require [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **ACP-FINISH** / **ACP-TOK-***.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §26 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — AUDIT-IDEAL Tier-3 rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-3.2 | §3 Intake | Product host intake parity (streaming + durable async default) | P2 | **Done** |
| AUDIT-IDEAL-28.1 | §28 Tier-3 | Durable async queue default beyond SQLite (DEBT-28-01) | P1 | **Done** |
| AUDIT-IDEAL-28.2 | §28 Tier-3 | Queue worker scaffold-default (`INCLUDE_QUEUE_WORKER`) | P1 | **Done** |
| AUDIT-IDEAL-28.3 | §28 Tier-3 | LKW hybrid daemon (CFG-14) | P4 | **Done** |
| AUDIT-IDEAL-28.4 | §28 Tier-3 | Business agents K.1/K.2 certification + deploy | P4 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

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
| H-APP.0.2 | H0 | **Author guide: environment vs agent** — what belongs in `applications/` vs `agents/`; forbidden patterns | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` or `guides/AGENT_CREATION_GUIDE.md` |
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

---

### Phase H — Interaction Surfaces (§18)

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

### Phase N — Application Environment & Deploy Scaffold (Tier-3)

**Canon:** §7.4.8–§7.4.10  
**Goal:** From agent POC to **docker-pushable** dedicated lab/product host in minutes — same ergonomics as `new-agent`, with isolated `.env.example`, manifest, and Docker.

**Prerequisite:** Phase L complete; Phase M.3 (`IntegrationProfile`) available.

**Delivery rule (this phase):** One step per iteration — implement → summarize → update docs → present next step (see **§6.1**).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| N.0 | Architecture & plan documented | **Done** | §7.4.8–§7.4.10 | This section + runtime canon (2026-05-30) |
| N.1 | `ApplicationManifest` + `AgentBinding` models | **Done** | §7.4.10 | `intergrax/applications/contracts/manifest.py` |
| N.2 | Manifest conformance harness + unit tests | **Done** | §7.4.10 | `intergrax/applications/_shared/wiring.py` |
| N.2.1 | Unified agent initialization (builders / factories / context) | **Done** | §7.4.10 | `ApplicationBuildContext`, `build_application_registry`; lab + legal migrated |
| N.2.2 | Strongly typed `AgentBinding.mount(AgentClass, factory=...)` | **Done** | §7.4.10 | `type[Agent]` + callable factory; `deserialize()` for scaffold strings only |
| N.3 | `python -m intergrax.scaffold new-application` (profile `lab`) | **Done** | §7.4.8 | `new_application.py`, `agent_catalog.py`, `cli.py`; lab templates + smoke |
| N.4 | Scaffold profile `product` (fastapi_core skeleton) | **Done** | §7.4.8 | `new_application_product.py`; FastAPI Core + auth stub + `/health`; `--agents` list |
| N.5 | Docker templates under `applications/<app>/docker/` | **Done** | §7.4.8 | Dockerfile + `.dockerignore` + `docker-compose.yml` + `build-docker.sh` / `.bat`; monorepo-root context |
| N.6 | Reference app `poc_template_application` (committed example) | **Done** | §7.4.8 | `applications/poc_template_application/`; README three-command quickstart; gate smoke |
| N.7 | Backfill `.env.example` on existing apps | **Done** | §7.4.8 | `lab_application`, `legal_application`, `research_application`, `poc_template_application` |
| N.8 | `guides/AGENT_CREATION_GUIDE.md` Step 4E (dedicated application) | **Done** | — | Step 4E + Appendix F cross-links; gate doc test |
| N.9 | Acceptance `test_scaffold_application` (gate) | **Done** | — | `test_scaffold_acceptance.py` — lab/product E2E, CLI profiles, docker scripts |
| N.10 | Optional `new-stack` (agent + application in one CLI) | **Done** | — | `intergrax/scaffold/new_stack.py`; gate test in `test_scaffold_acceptance.py` |

#### N — Step-by-step implementation sequence

Execute **strictly in order**; do not skip ahead without completing acceptance for the current step.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | N.1 | Add `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` (Pydantic) | Unit tests pass; no scaffold yet |
| 2 | N.2 | Add `applications/_shared/conformance.py` (or mirror integrations pattern) | Manifest load + minimal registry build test |
| 3 | N.3 | Implement `new_application.py` + `lab` profile templates | `uv run python -m intergrax.scaffold new-application test_lab --profile lab --agents echo` creates tree; smoke test green |
| 4 | N.3b | Wire `build_parser()` subcommand; post-create hints (uvicorn, pytest, docker) | CLI prints next commands; gate test added (N.9 partial) |
| 5 | N.5 | Add Docker/docker-compose + build scripts to scaffold | `applications/<app>/docker/build-docker.sh` (or `.bat`) builds image from repo root |
| 6 | N.6 | Commit `applications/poc_template_application/` from scaffold | README three-command quickstart verified |
| 7 | N.7 | Add per-app `.env.example` to legal, research, lab | Vars match each `settings.py`; no secrets committed |
| 8 | N.4 | Add `product` profile to scaffold | **Done** — `test_scaffold_product_application.py`; FastAPI Core + `/health` |
| 9 | N.8 | Update agent guide Step 4E | **Done** — scaffold lab/product, Docker scripts, three-command quickstart |
| 10 | N.9 | Full acceptance + `pytest -m gate` | **Done** — runtime E2E + `test_scaffold_acceptance.py` |

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
- Runtime sandbox (Tier-1) changes — only document distinction (§7.4.9)

#### Tier-3 application layer — readiness (2026-05-30)

**Status: ready** to generate new applications via scaffold. Checklist: [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md).

| Track | ID | Status | Notes |
|-------|-----|--------|-------|
| Engine | N.1–N.2.2 | **Done** | manifest, `build_application_registry`, conformance |
| Scaffold | N.3–N.4, N.10 | **Done** | `lab` + `product` + `new-stack` |
| Deploy | N.5–N.7 | **Done** | Docker scripts, `BUILD_AND_DEPLOY`, `.env.example` |
| Docs + gate | N.8–N.9 | **Done** | Step 4E, `test_scaffold_acceptance`, legal/research/lab manifest tests |
| Hardening | A.1–A.2 | **Done** | `test_legal_manifest_wiring`, tool_wiring assertions on scaffold |
| Optional CI Docker | B.1 | **Done** | `tests/integration/applications/test_poc_template_docker_build.py` (not in gate) |
| Product maturity | — | **Reference** | `legal_application` chat routes — extend scaffold `product` manually |

**Verify:**

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest -m gate -q
```

---

## Phase H-APP-DOC — Application interaction & orchestration authoring (Band 2ar — docs)

**Status:** **Done** (2026-06-09) — architecture canon §23; cross-refs to ORCHESTRATION §55, REASONING §9.4, NEXUS_EXECUTION_FLOW §3.1  
**Prerequisites:** Phase H-APP **Done** · Phase ORCH-STRAT **Done** · Phase COG-DOC **Done**  
**Goal:** Close authoring gaps for flexible Tier-3 postures (daemon, reactive, background) and multi-agent configuration without runtime changes.

**ADR:** [`ADR-FLOW-004`](../adr/entries/2026-06-09/ADR-FLOW-004.md) for `trigger_capabilities` (H-APP-DOC.2 / ORCH-CONFIG.2 **Done**). Authoring-only items need no ADR.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-DOC.1 | **Architecture §23** — posture catalog, routing matrix, scenario recipes | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` §23 |
| H-APP-DOC.2 | **`ApplicationGraphSpec.trigger_capabilities`** — optional seed guard (code) | **Done** | Medium | ORCH-CONFIG.2 · ADR-FLOW-004 · `test_graph_spec_to_plan.py` |
| H-APP-DOC.3 | **`intergrax/applications/USAGE.md` §** — orchestration configuration (ORCH-CONFIG / §56.13) | **Done** | Medium | Posture presets + harness proof links |
| H-APP-DOC.4 | **Scaffold `new-application` product** — interaction intake + scheduler optional wire | **Done** | Low | `INCLUDE_INTERACTIONS` / `INCLUDE_SCHEDULER`; legal host reference |

**Explicitly out of scope:** Nexus runtime fork; new coordination patterns (ORCH-5); COG-3 classifier implementation (tracked under ORCH-CONFIG.1 / COG-3.*).

**Canonical platform cases:** [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §56 · implementation register [`plan/ORCHESTRATION.md`](../plan/ORCHESTRATION.md) Phase **ORCH-CONFIG**.

---

## Phase H-APP-WIRING — Tier-3 execution surface parity (Band 2aw — planned)

**Status:** **Done** (2026-06-09) — **6/6 Done** · CFG host parity closeout (2026-06-09)  
**Audit source:** [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §59 · [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §23.7–§23.8 · FLOW-GAP-17–20  
**Prerequisites:** Phase H-APP **Done** · ORCH-6 **Done** · FLOW-CTL **Done** · REL-ADV **Done**  
**Goal:** Close **docs ↔ code discrepancies** where platform capabilities exist in Tier-1 but product hosts expose only sync `/run` — without Nexus forks.

**Priority ladder:** **Band 2aw** — recommended harness band after §6.1 gate maintenance (before Band 3 §6.3).

| ID | Gap / T3-GAP | Deliverable | Status | Priority | Acceptance |
|----|--------------|-------------|--------|----------|------------|
| H-APP-WIRING.1 | T3-GAP-01, T3-GAP-02 | Scaffold `INCLUDE_TASK_CONTROL` → optional `mount_harness_task_routes` + `apply_reliability_task_defaults` in `new-application` / `new-stack` | **Done** | **Critical** | `task_control_wiring.py` · `test_harness_task_control_wiring.py` |
| H-APP-WIRING.2 | T3-GAP-03, T3-GAP-04 | Adopt scheduler + task control on legal + research + poc_template reference hosts | **Done** | High | `legal_application` / `research_application` / `poc_template_application` factories |
| H-APP-WIRING.3 | T3-GAP-05, FLOW-GAP-18 | Optional `QueuedNexusExecutionAdapter` via `queue_worker_wiring.py` + `INCLUDE_QUEUE_WORKER` | **Done** | High | Legal host; scaffold env flags |
| H-APP-WIRING.4 | FLOW-GAP-20, CFG-14 | LKW hybrid daemon — explicit deferral in `local_workspace_application/ARCHITECTURE.md` | **Done** | Medium | §6.3 product backlog unchanged |
| H-APP-WIRING.5 | T3-GAP-01–04 | Task control + enricher + scheduler on assistant + dispute_sim + LKW hosts | **Done** | High | `intergrax_assistant_application` / `dispute_sim_application` / `local_workspace_application` factories |
| H-APP-WIRING-DOC.1 | — | Sync architecture §23.7–§23.8 + ORCH §59.2 host matrix | **Done** | Low | This phase closeout |

**Explicitly out of scope:** Nexus runtime changes; K.1/K.2; new queue transport.

**Cross-plan:** H-APP-WIRING.1 ↔ ORCH-6.5 · FLOW-CTL.6 · REL-ADV.7.

---

## Phase H-APP-CON — Application Environment Architecture canon (APP-CON)

**Status:** **In progress** (2026-06-11) — architecture §24–§51 frozen; implementation **3/10 APP-CON** + **5/9 APP-PROD** Done — see [Master backlog](#master-implementation-backlog-app-unified)  
**Prerequisites:** Phase H-APP **Done** · H-APP-DOC **Done** · H-APP-WIRING **Done**  
**Goal:** Deliver **symmetric authoring canon** to ACP for Tier-3 — contracts, facades, hooks, checklists — without a new domain pair or Nexus fork.

**ADR:** no ADR needed for documentation-only tranche; **ADR-APP-001** recommended when mounting `ApplicationHost` into production pipeline (APP-CON-1 **Done**).

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-CON-DOC.1 | Architecture §24–§51 + TOC + fidelity matrix (this plan) | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` + §Architecture fidelity matrix |
| H-APP-CON-DOC.2 | Hub § Application in harness environment | **Done** | High | `intergrax_runtime_architecture.md` |
| H-APP-CON-DOC.3 | Cross-ref ACP §39 → TIER3 §39 canonical home | **Done** | Low | ACP §39.8 pointer |
| APP-CON-1..8 | Host contracts — see [APP-CON master](#app-con--host-contracts-architecture-25-32--42--48) | **Partial** | **Critical** | 1,2,4 Done |
| APP-PROD-1..9 | Release gates — see [APP-PROD master](#app-prod--release-gates-architecture-40--46) | **Partial** | High | 1–5 Done |
| APP-CON-DX.* | Author + audit DX | Planned | Medium | §31 · §45 · §47 |

**Explicitly out of scope:** `Application.on_next_orchestration_step()`; new domain pair; Nexus runtime changes for product-specific orchestration.

**Rejected (documented in architecture §28.2):** cloning ACP step loop at Tier-3.

---

## Phase H-APP-EVOL — Runtime evolution and governance (APP-EVOL)

**Status:** **In progress** (2026-06-11) — architecture §49 documented; implementation APP-EVOL-1..7 **Planned**  
**Prerequisites:** H-APP-CON architecture **Done** · V-ALG.3 agent lifecycle **Done**  
**Goal:** Close operational gaps for large-scale Tier-3 — versioning, migration, capability sunset, agent certification, recovery contract, environment diff, application packaging — without Nexus or profile primitive changes.

**ADR:** no ADR needed for §49 documentation tranche; **ADR-APP-002** recommended when `EnvironmentSnapshot` becomes mandatory on STRICT intake (APP-EVOL-1).

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-EVOL-DOC.1 | Architecture §49 Runtime Evolution and Governance | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` |
| APP-EVOL-1 | `EnvironmentSnapshot` + intake `profile_snapshot_id` | **Done** | **Critical** | `test_environment_snapshot_wiring.py` |
| APP-EVOL-2 | `ApplicationMigration` schema + CI validator | **Done** | High | `check_application_migrations.py` |
| APP-EVOL-3 | `CapabilityAlias` + deprecation routing | **Done** | High | `test_capability_alias_wiring.py` |
| APP-EVOL-4 | `AgentCertification` + STRICT roster gate | Planned | High | non-PRODUCTION blocked |
| APP-EVOL-5 | `ApplicationRecoveryContract` on profile | Planned | High | product ARCHITECTURE template |
| APP-EVOL-6 | `ApplicationEnvironmentDiff` + `doctor diff-app` | Planned | Medium | pre-deploy CI diff |
| APP-EVOL-7 | `ApplicationPackage` + dependency resolver | Planned | Medium | `new-stack` emits package manifest |

**Explicitly out of scope:** marketplace UI; Nexus fork; Tier-3 cognition loop.

---

## Phase H-APP-OPS — Platform operations canon (APP-OPS) — freeze tranche

**Status:** **Done** (2026-06-11) — architecture §50 documented; APP-OPS-1..4 **Planned**  
**Prerequisites:** H-APP-EVOL §49 **Done** · V-CG.1–3 capability graph **Done**  
**Goal:** Close reference-platform gaps — capability graph at environment scope, application ownership, health scoring, application/environment registry — **without** changing frozen primitives (Nexus, ApplicationHost, profile, graph spec, envelope, hooks).

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| H-APP-OPS-DOC.1 | Architecture §50 Platform Operations Canon | **Done** | **Critical** | `architecture/TIER3_APPLICATION_ENVIRONMENT.md` |
| H-APP-OPS-DOC.2 | §49.2.4 typed migrations (Profile/Graph/Envelope) | **Done** | High | sub-migration schemas in §49 |
| APP-OPS-1 | Env capability graph + blast radius STRICT gate | **Done** | **Critical** | `check_capability_graph_strict_deploy.py` |
| APP-OPS-2 | `ApplicationOperationalOwnership` + APP-PROD | **Done** | High | `check_application_ownership.py` |
| APP-OPS-3 | `EnvironmentHealthScore` + `doctor health-app` | Planned | High | release score artifact |
| APP-OPS-4 | `ApplicationRegistry` + `EnvironmentRegistry` + CLI | Planned | Medium | `apps list` / `envs list` |
| APP-EVOL-2b | Typed migration validators | **Done** | High | `migration_wiring.py` per primitive |

**Freeze declaration:** Tier-3 **structural architecture** is complete at §51. Further work is APP-* implementation only — no new composition primitives without ADR.

---

## Phase H-APP-FREEZE — Cross-document governance consistency audit

**Status:** **Done** (2026-06-11)  
**Goal:** Verify semantic alignment between Tier-3, ACP, UAEP, IDEAL — no duplicate capability/registry/ownership/health definitions before architecture freeze.

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| H-APP-FREEZE-1 | `guides/GOVERNANCE_CONSISTENCY_AUDIT.md` | **Done** | Five audit questions answered |
| H-APP-FREEZE-2 | TIER3 §51 + ACP §19 cross-refs | **Done** | Canonical ownership matrix |
| H-APP-FREEZE-3 | §22 GovernanceProfile description fix | **Done** | Flags ≠ ownership |

**Outcome:** No structural conflicts. Glossary bans `CapabilityRegistry`. Architecture freeze **approved**.

---
