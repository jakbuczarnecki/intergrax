# TIER3_APPLICATION_ENVIRONMENT — production gates (§40+)

**Parent hub:** [`TIER3_APPLICATION_ENVIRONMENT.md`](../TIER3_APPLICATION_ENVIRONMENT.md)

# 40. Production Reliability, Safety, and Release Gates (Tier-3)

Symmetric to ACP §40 — **host environments** that run mutating workloads.

## 40.1 Host readiness dimensions

| Dimension | Requirement |
|-----------|-------------|
| Wiring | `build_harness_host_runtime` — no ad-hoc Nexus |
| Identity | `IdentityProfile` enforced on prod routes |
| Execution mode | `STRICT` in production |
| Reliability | `ReliabilityProfile` + task checkpoints when long-running |
| Observability | `ObservabilityProfile` + trace persistence |
| Roster | `EnvironmentSkillToolConsistencyCheck` passes |
| Org / compliance | Envelope + eval golden scenarios when UC-A7 |
| Deploy triad | Docker + `BUILD_AND_DEPLOY.md` + gate test |

## 40.2 APP-PROD gate register

| ID | Deliverable | Status | Command / test |
|----|-------------|--------|----------------|
| APP-PROD-1 | `check_application_production_gates.py` — no ad-hoc Nexus, harness runtime | **Done** | `python scripts/gates/check_application_production_gates.py` |
| APP-PROD-2 | Reference hosts use `build_harness_host_runtime` exclusively | **Done** | H-APP-WIRING |
| APP-PROD-3 | `ApplicationHost` mounted when provided | **Done** | `test_application_host_wiring` |
| APP-PROD-4 | Manifest conformance | **Done** | `test_manifest_conformance` |
| APP-PROD-5 | Deploy triad | **Done** | `test_application_deploy_triad` |
| APP-PROD-6 | `check_environment_state_usage` — hooks use `app_env_state.v1` | **Done** | `check_environment_state_usage.py` · `environment_state_usage_wiring.py` |
| APP-PROD-7 | `check_budget_enforcement` — COST profile on STRICT product hosts | **Done** | `check_budget_enforcement.py` |
| APP-PROD-8 | `check_workspace_cleanup` — factory lifespan cleanup hooks | **Done** | `check_workspace_cleanup.py` · `build_factory_lifespans` |
| APP-PROD-9 | Gate test + CI `gate-governance-tier` | **Done** | `test_check_application_production_gates.py` |

## 40.3 Mutating product checklist

Before claiming production-ready for mutating hosts:

1. `execution_mode=strict`
2. `ReliabilityProfile` idempotency + checkpoints enabled
3. `CriticProfile` for high-risk capabilities
4. `mount_harness_task_routes` when HITL/long-running required
5. Product `ARCHITECTURE.md` documents §23.7 gaps closed or deferred

---

# 41. Composition Primitives — Separation Matrix

Normative mapping — **do not conflate** these primitives:

| Primitive | Layer | Answers | Does NOT |
|-----------|-------|---------|----------|
| **`ApplicationGraphSpec`** | Declarative topology | Which agents, in what order/parallelism, edges | Domain reasoning; per-step tool calls |
| **`ApplicationHost`** | Imperative reactions | Dynamic block/modify/escalate at Nexus events | Replace graph; cognitive loop |
| **`OrganizationalPolicyEnvelope`** | Rules / simulation | Org-wide channels, playbooks, tool denies | Per-agent factory logic |
| **`AgentBinding`** | Per-agent wiring | Implementation class, capability, slices, `org_role_id` | Orchestration topology |
| **`ApplicationEnvironmentProfile`** | Harness slices (§22.1 flat · §22.6 bundles) | Catalogs, modes, observability, cost, reliability | Business rules in code; not a second composition root |
| **`ShadowWorkspaceProfile` / `SandboxProfile`** | Isolation | Safe experiments / code exec | Agent selection |
| **`NexusLoop`** | Tier-1 OS | Execute Task graph with policy | Product-specific forks |

```text
Topology     → ApplicationGraphSpec (+ OrchestrationProfile)
Rules        → OrganizationalPolicyEnvelope + PolicyRulesProfile
Per-agent    → AgentBinding → merge_environment()
Reactions    → ApplicationHost.on_hook()
Catalogs     → ApplicationEnvironmentProfile → CapabilityBundle (§22.6) or flat sub-profiles (§22.1)
Cognition    → Agent.on_next_step() ONLY
```

---

# 42. ApplicationEnvironmentState (Typed Host State)

**Contract:** `intergrax/applications/contracts/environment_state.py` · schema **`app_env_state.v2`** on wire key **`app_env_state.v1`**.

Hooks receive `HookContext.runtime_state: dict`. Application authors MUST use the typed model — not ad-hoc keys.

## 42.1 Core fields

```text
ApplicationEnvironmentState:
    schema_version: app_env_state.v2
    app_id, profile_id, profile_snapshot_id
    execution_mode
    task_id, run_id, graph_id | null
    phase: EnvironmentTaskPhase
    health: EnvironmentHealthStatus
    organization_id | null
    policy_overlays: PolicyOverlayState
    hitl: HitlEscalationState
    budget: ActiveBudgetState
    shadow_workspace: WorkspaceIsolationRef | null
    sandbox_session: SandboxIsolationRef | null
    pending_notifications: list[PendingNotification]
    custom: dict                              # small product extensions only
```

## 42.2 Nested models

| Model | Purpose |
|-------|---------|
| `PolicyOverlayState` | Org id, role, scenario, playbook ids, tool denies, prompt overlays |
| `HitlEscalationState` | `pending`, `ticket_id`, `escalation_reason`, `awaiting_role` |
| `ActiveBudgetState` | Agent/env token totals, limits, warn/emitted/exceeded, `last_reaction` |
| `WorkspaceIsolationRef` / `SandboxIsolationRef` | Active isolation handles + paths |
| `PendingNotification` | Queued notify channel + template for host reactions |

## 42.3 Persistence rules

| State class | Scope | Persistence |
|-------------|-------|-------------|
| `app_env_state.v1` | Single **Task** lifecycle | MODIFY merges across hooks; cleared on new task |
| Agent cognition `acp.state.v1` | Agent run | ACP checkpoint — separate plane |
| Artifacts §48 | Task + retention policy | Filesystem / object store |
| Trace / summary | Ops | OBS spine + `ApplicationRunSummary` |

**Rules:** no secrets in `custom`; no unbounded lists; cross-task workflow state → task memory or external store.

## 42.4 Helpers

- `seed_application_environment_state(...)` — intake bootstrap
- `ApplicationEnvironmentState.from_runtime_state(ctx.runtime_state)`
- `state.patch_runtime_state()` → `HookResult.modified_payload`

**Done APP-CON-3:** `ApplicationEnvironmentStateMiddleware` auto-updates `phase`, `budget`, HITL fields on hook context (`application_environment_state_middleware.py`).

---

# 43. Budget Reactions and Token Governance

Symmetric to ACP §25.5 — **application configures**, **harness enforces**, **agents read**. Full agent-side detail: ACP §25.4–§25.5.

## 43.1 End-to-end runtime flow

```text
Tier-3 config (CostProfile + AgentBinding.budget_slice + budget_reaction)
    → materialize_runtime_config / merge_environment
    → ResolvedBudgetLimits on AgentStepContext (ACP-TOK-1)
    → each LLM call meters tokens (LLM adapters + §25.4 rollups)
    → HarnessKernel pre-LLM check (ACP-TOK-2):
         if tokens_total >= limit * warn_threshold_ratio → BUDGET_THRESHOLD event + notify
         if hard limit exceeded → apply BudgetExceededReaction
    → ApplicationEnvironmentState.budget updated (APP-CON-3)
    → host notify / custom_hook / HITL / abort / degrade_model (ACP-TOK-3)
    → Plane A ApplicationRunSummary totals + Plane B step records
```

## 43.2 Configuration surfaces (Tier-3)

| Surface | Field | Scope |
|---------|-------|-------|
| Environment ceiling | `CostProfile.max_total_tokens` | Whole task / graph (`RunBudget` Nexus) |
| Per-agent cap | `AgentBinding.budget_slice` | Single agent run |
| Reactions | `CostProfile.budget_reaction` | Threshold + exceed behavior |
| Enforcement | `AgentBudgetSlice.enforcement` | `hard` \| `advisory` |

**Merge order:** platform default → `cost_profile` → `budget_slice` → request overrides (STRICT denies widen).

## 43.3 BudgetReactionProfile (normative)

```text
BudgetReactionProfile:
    on_agent_limit_exceeded: abort | hitl | degrade_model | notify_only | custom_hook
    on_environment_limit_exceeded: abort | hitl | degrade_model | notify_only | custom_hook | pause_graph
    notify_channels: list[in_app | webhook | slack | email | trace_only]
    warn_threshold_ratio: float = 0.80
    custom_hook_id: str | null
    user_message_template: str | null
```

## 43.4 Soft vs hard caps

| Kind | Detection | Kernel | Host (Tier-3) |
|------|-----------|-----------------|---------------|
| **Soft** | usage ≥ limit × ratio | `BUDGET_THRESHOLD` event | `notify_channels`; update `budget.warn_emitted` |
| **Hard agent** | agent scope ≥ limit, `enforcement=hard` | Block LLM; `on_agent_limit_exceeded` | HITL ticket / webhook / `custom_hook_id` |
| **Hard environment** | env scope ≥ limit | Block graph LLM; `on_environment_limit_exceeded` | May `pause_graph` + operator alert |
| **Advisory** | limit set, `enforcement=advisory` | Meter only | Agent soft strategy in `on_next_step` |

## 43.5 Reaction semantics (kernel + host)

| Reaction | Kernel effect | Host / operator surface |
|----------|---------------|-------------------------|
| **`abort`** | `BUDGET_EXCEEDED`, terminal `budget_exceeded` | Error + `user_message_template` |
| **`hitl`** | `pause_hitl` / Nexus HITL runner | `HitlEscalationState` §42 |
| **`degrade_model`** | `StepLLMRouter` cheapest allowed model | Trace warning |
| **`notify_only`** | Continue if advisory; always emit events | Slack/webhook/in_app via integration slugs |
| **`custom_hook`** | Emit payload to host registry | Billing, paging, CRM — **no vendor SDK in Tier-2** |
| **`pause_graph`** | Environment exceed only — freeze graph | Task status + summary |

## 43.6 Acceptance tests (gates)

| Test | Asserts | Gate |
|------|---------|------|
| `test_budget_threshold_event` | Soft warn at 80% | ACP-TOK-2 |
| `test_hard_cap_blocks_llm` | No LLM after exceed | ACP-TOK-2 |
| `test_budget_reaction_hitl` | HITL pause on agent exceed | ACP-TOK-3 |
| `test_budget_custom_hook` | Host callback invoked | ACP-TOK-3 |
| `test_environment_cap_pause_graph` | Graph stops on env exceed | ACP-TOK-3 |
| `check_application_production_gates` | Host wiring + manifest | APP-PROD-1 |
| `check_budget_enforcement` | STRICT product COST + `budget_slice` | APP-PROD-7 |

## 43.7 Implementation status (honest)

| ID | Deliverable | Status |
|----|-------------|--------|
| Contracts | `BudgetReactionProfile`, `AgentBudgetSlice` | **Done** |
| Metering | `invocation_usage` rollups | **Done** ACP-TOK-1 |
| Kernel enforce + reactions | `HarnessKernel` pre-LLM | **Done** ACP-TOK-2 · ACP-TOK-3 |
| Host notify + hooks | Tier-3 wiring | **Done** ACP-TOK-3 |
| Product gate | `check_budget_enforcement` | **Done** APP-PROD-7 |
| Nexus `RunBudget` | Environment cap | **Partial** COST-1 |

**Production claim:** mutating STRICT product hosts MUST declare `budget_reaction` + per-agent `budget_slice` (APP-PROD-7).

**Anti-pattern BUD-AP-01:** Hardcoded limits in `on_next_step`. **Correct:** `budget_slice` + `budget_reaction`.

---

# 44. Scenario Test Matrix (Tier-3)

Minimum verification before claiming host maturity. Map to §23.5 recipes and §35 UC-A*.

| Scenario | Posture | Required tests | Key assertions |
|----------|---------|----------------|----------------|
| **Reactive single-agent** | HTTP `/run` | Unit: manifest conformance; integration: `run_task` | `TaskResult` completed; Plane A summary |
| **Always-on daemon** | `serve()` / factory lifespan | Smoke: health + `/run` | Process boots; scheduler if enabled |
| **Scheduled / queue** | `INCLUDE_QUEUE_WORKER` | Integration: enqueue → worker | Async completion notification |
| **Hybrid** | daemon + queue | Product ARCHITECTURE + integration | Background + interactive paths |
| **Multi-agent graph** | `graph_spec` | `test_lab_graph_spec` pattern | Node order / parallel batches in trace |
| **Virtual org** | `organizational_policy` | UC-11 golden (ACP-ORG-5) | `PolicyVerdictRecord`; denied tools blocked |
| **Simulation** | dispute_sim / scenario bindings | Graph + scenario metadata | Scenario playbook overlay applied |
| **Mutating prod** | STRICT + reliability | ACP-PROD + APP-PROD §46 | Idempotency + checkpoint on host |
| **ApplicationHost hook** | any | `test_application_host_wiring` | Middleware mounted; BLOCK works |
| **Budget exceed** | cost_profile | ACP-TOK-2 · ACP-TOK-3 · APP-PROD-7 | `BUDGET_EXCEEDED` + reaction path |

**Gate commands:**

```bash
python scripts/maintenance/check_tier3_scenario_matrix.py
uv run pytest tests/unit/applications/test_tier3_scenario_matrix.py -m tier3_scenario -q
uv run pytest -m gate -q
```

**Registry:** `intergrax/applications/_shared/tier3_scenario_matrix_wiring.py` maps each reference host package to minimum §44 scenarios and UC-A* evidence paths under `tests/unit/applications/`.

---

# 46. Production Readiness Acceptance Criteria

A Tier-3 host MAY be labeled **production-ready** only when **all** mandatory rows pass for its posture class.

## 46.1 Mandatory (every product host)

| # | Criterion | Evidence |
|---|-----------|----------|
| P1 | `ApplicationManifest` + full `ApplicationEnvironmentProfile` on manifest | `test_manifest_conformance` |
| P2 | `build_harness_host_runtime()` — no ad-hoc `NexusLoop(...)` | Code review / APP-PROD-1 |
| P3 | `wire_application_environment()` — no `getattr` on manifest | `check_harness_no_getattr` |
| P4 | All surfaces → `UnifiedTaskRunner.run_task()` | Factory + router review |
| P5 | `execution_mode=strict` in production profile | `environment_profile.py` |
| P6 | `IdentityProfile` matches deployed auth | Integration test or manual runbook |
| P7 | `EnvironmentSkillToolConsistencyCheck` passes | Wiring logs / unit test |
| P8 | Deploy triad (Docker, `BUILD_AND_DEPLOY.md`, `.env.example`) | `test_application_deploy_triad` |
| P9 | Business logic only in Tier-2 agents | `check_agent_registry_bypass` |
| P10 | §23.7 host gaps closed **or** documented in product `ARCHITECTURE.md` | Doc link |

## 46.2 Required when capability applies

| Capability | Additional criteria |
|------------|---------------------|
| Long-running / HITL | `ReliabilityProfile` + `mount_harness_task_routes` + checkpoint store |
| Multi-agent | `graph_spec` or documented pipeline token + `ApplicationRunSummary` test |
| Interaction intake | `wire_interaction_intake_service` + signature tests |
| Virtual org (UC-A7) | `OrganizationalPolicyEnvelope` + eval golden zero `POLICY_DENIED` on happy path |
| `ApplicationHost` hooks | APP-CON-1 middleware mounted + hook unit test |
| Mutating tools in STRICT | ACP-PROD gates on agents + host idempotency store |
| Budget-sensitive | `budget_reaction` + per-agent `budget_slice`; APP-PROD-7 gate on STRICT hosts |

## 46.3 Maturity score (architecture audit)

| Dimension | Target | Current (2026-06-14) |
|-----------|--------|----------------------|
| Architecture completeness | 10/10 | **10/10** — APP-CON §24–§48 + evolution §49 + ops §50 |
| Hook runtime wiring | 10/10 | **10/10** — APP-CON-1 · APP-CON-5 Done |
| Budget / prod gates | 10/10 | **10/10** — APP-PROD-1..9 **Done** · ACP-TOK-1..3 · ACP-TOK-CI **Done** |
| Evolution / governance | 10/10 | **10/10** — APP-EVOL-1..7 **Done** · §49.2.4 typed migrations |
| Platform operations | 10/10 | **10/10** — APP-OPS-1..4 **Done** · health score · registry CLI |
| **Overall production readiness** | — | **~9.5/10** reference platform; enterprise marketplace/distribution **P4** |
| **Architecture freeze readiness** | — | **Architecturally Mature** — §24–§51 + APP-* **Done**; P4 = marketplace UI + semver on graph/envelope models |

---

# 47. Developer Mental Model

**“What do I implement for environment type X?”** — five recipes. Cognition stays in agents only.

## 47.1 Minimal lab application

| Implement | Do not implement |
|-----------|------------------|
| `manifest.py` + `ApplicationEnvironmentProfile.lab_defaults()` | Nexus subclass |
| `host/factory.py` → `build_harness_host_runtime` | `on_next_step` in host |
| `AgentBinding.mount(EchoAgent)` | Business rules in factory |
| Optional `HarnessApplication` for quick test | Org envelope |

**Files:** `manifest.py`, `host/environment_profile.py`, `host/factory.py`, `host/main.py`, `.env.example`

## 47.2 Product application (single/multi agent)

| Implement | Do not implement |
|-----------|------------------|
| Full `ApplicationEnvironmentProfile` (STRICT, OBS, REL) | Ad-hoc `NexusLoop(` |
| Roster + factories per agent | Multi-agent loops in Tier-3 |
| `graph_spec` **or** explicit API capabilities | Hidden agent routing |
| HTTP/MCP routes → `UnifiedTaskRunner` | Direct `agent.run()` from routers |
| Deploy triad | |

**Files:** above + `serving/fastapi_router.py`, `docker/`, `BUILD_AND_DEPLOY.md`, product `ARCHITECTURE.md`

## 47.3 Virtual organization

| Implement | Do not implement |
|-----------|------------------|
| `OrganizationalPolicyEnvelope` on profile | `if org ==` in agents |
| `AgentBinding.org_role_id` per role | Duplicate compliance in Tier-2 |
| Policy YAML under `host/policy/rules/` | |
| Eval golden scenarios (UC-A7) | |

## 47.4 Simulation / scenario host

| Implement | Do not implement |
|-----------|------------------|
| `ApplicationGraphSpec` + `scenario_bindings` | Custom orchestration loop |
| `capability=*.pipeline` on API | |
| `dispute_sim`-style reference patterns | |

## 47.5 Mutating production host

| Implement | Do not implement |
|-----------|------------------|
| Everything in §47.2 + §46.1 mandatory | Ship without `budget_reaction` + `budget_slice` (APP-PROD-7) |
| `ReliabilityProfile` idempotency + checkpoints | |
| `CriticProfile` for high-risk caps | |
| `budget_reaction` when cost-sensitive | |
| `mount_harness_task_routes` for HITL | |
| Pass `check_application_production_gates.py` | |

**Rule of thumb:** if it **thinks**, it belongs in **`agents/`**. If it **composes, constrains, or reacts**, it belongs in **`applications/`** profile/manifest/hooks.

---

# 48. Application Artifacts

**Contract:** `intergrax/applications/contracts/application_artifacts.py`

Artifacts are **first-class outputs** of application environments — linked to `task_id`, `run_id`, `graph_id`, with provenance and retention.

## 48.1 Reference types

| Type | Model | Typical source |
|------|-------|----------------|
| Application | `ApplicationArtifactRef` | Business outcome webhooks, exports |
| Shadow workspace | `WorkspaceArtifactRef` | `ShadowWorkspace.list_artifacts()` |
| Sandbox | `SandboxArtifactRef` | `sandbox.exec` outputs |
| Rollup | `RunArtifactBundle` | Attached to `ApplicationRunSummary` metadata |

## 48.2 Common fields

```text
artifact_id, uri, size_bytes, sha256
task_id, run_id?, graph_id?
owner_app_id, tenant_id
security_class: public | internal | confidential | restricted
visibility: task_only | application | tenant | operator
retention: retain_hours, delete_on_task_complete, archive_to_object_store
provenance: application | shadow_workspace | sandbox | tool
```

## 48.3 Metadata keys

| Key | Content |
|-----|---------|
| `application_run_summary.v1` | Plane A rollup §26 |
| `run_artifact_bundle.v1` | `RunArtifactBundle` §48 |

## 48.4 Lifecycle

```text
produce → classify → attach to task metadata → expose in summary → retain/purge per policy
```

**Rule:** operators discover artifacts via summary + bundle — not by scanning host filesystem ad hoc.

---

# 49. Runtime Evolution and Governance

Operational lifecycle for Tier-3 environments at scale — **versioning, migration, capability sunset, agent promotion, recovery, diff, packaging**. This chapter does **not** introduce a new cognition loop or Nexus fork; it defines how **declarative** application artifacts evolve and how **hosts** react when reality diverges from config.

**Design principle:** configuration is immutable-at-a-point-in-time; **snapshots** + **migrations** make change auditable. Runtime always executes against a **resolved snapshot**, not “latest YAML on disk” in STRICT production.

```text
Author edits manifest / profile / graph / envelope
    → version bump + migration script (when breaking)
    → EnvironmentSnapshot materialized at deploy / task intake
    → Nexus executes against snapshot
    → Recovery / diff / audit use same snapshot ids
```

**Cross-domain:** agent cognition versioning → ACP §25 · capability semver → UAEP §42.27 · checkpoint/resume → [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) §33.3 · agent lifecycle evaluator → `intergrax/runtime/architecture/agent_lifecycle_governance.py`.

---

## 49.1 Environment Versioning

### 49.1.1 Version surfaces (normative)

| Artifact | Version field | Semantics |
|----------|---------------|-----------|
| **`ApplicationManifest`** | `version: semver` | Deployable application package release |
| **`ApplicationEnvironmentProfile`** | `spec_version: str` | Serialized profile shape for UI round-trip (DX-7.2) |
| **`ApplicationGraphSpec`** | `graph_version: semver` (**P4 backlog**) | Migration schema supports versions; model field not yet on `ApplicationGraphSpec` |
| **`OrganizationalPolicyEnvelope`** | `envelope_version: semver` (**P4 backlog**) | Migration schema supports versions; model uses `schema_version` today |
| **`ApplicationEnvironmentState`** | `profile_snapshot_id` | Active resolved profile fingerprint for a Task |
| **Wire contracts** | `schema_version` | e.g. `app_env_state.v2`, `run_artifact_bundle.v1` |

### 49.1.2 EnvironmentSnapshot (**Done** · APP-EVOL-1)

Immutable materialization of everything Nexus needs for one deploy or one Task intake:

```text
EnvironmentSnapshot:
    snapshot_id: str                    # stable hash or uuid
    app_id: str
    app_version: semver
    profile_snapshot_id: str
    manifest_digest: sha256
    graph_spec_digest: sha256 | null
    org_envelope_digest: sha256 | null
    roster_digest: sha256               # AgentBinding[] resolved
    captured_at: datetime
    captured_by: deploy | intake | manual_export
```

**Rules:**

- STRICT production Tasks SHOULD record `profile_snapshot_id` on `ApplicationEnvironmentState` (§42).
- Lab MAY run without snapshot persistence — product hosts MUST NOT.
- Snapshot is the **unit of replay** for simulation and post-incident audit.

### 49.1.3 ApplicationVersion

Logical release of a Tier-3 host — ties together manifest semver, container image tag, and optional changelog:

```text
ApplicationVersion:
    app_id: str
    version: semver
    git_ref: str | null
    image_tag: str | null
    changelog_ref: str | null
    compatible_runtime: str            # harness baseline, e.g. "1.0.0"
```

**Status:** `ApplicationManifest.version` **Done**; `EnvironmentSnapshot` **Done** (`APP-EVOL-1` · ADR-APP-002); `ApplicationPackage` closure **Done** (`APP-EVOL-7` · `package_wiring.py`).

---

## 49.2 Environment Migration

### 49.2.1 ApplicationMigration

Declarative description of how to move from snapshot A → B:

```text
ApplicationMigration:
    migration_id: str
    from_app_version: semver_range
    to_app_version: semver
    steps: list[MigrationStep]
    rollback_supported: bool
```

```text
MigrationStep:
    target: profile | graph_spec | org_envelope | roster | hooks
    action: transform | replace | validate_only
    script_ref: str                      # e.g. migrations/2026_06_profile_v2.py
    breaking: bool
```

### 49.2.2 What migrates (by primitive)

| Primitive | Typical change | Migration strategy |
|-----------|----------------|-------------------|
| **`ApplicationEnvironmentProfile`** | New sub-profile field, default change | Transform script + `spec_version` bump |
| **`ApplicationGraphSpec`** | Node rename, edge change | Graph migration + golden trace replay |
| **`OrganizationalPolicyEnvelope`** | Playbook/tool deny change | Envelope version + eval golden refresh |
| **`AgentBinding`** | Capability rename, agent swap | Roster migration + alias period (§49.3) |
| **Hooks** | New HookPoint behavior | Host code deploy — not data migration |

### 49.2.3 EnvironmentUpgrade flow (runtime)

```text
1. Operator bumps ApplicationManifest.version
2. CI runs migration validators + scenario matrix §44
3. Deploy new image → factory builds with new profile
4. On first Task intake after deploy:
     capture EnvironmentSnapshot
     seed app_env_state with profile_snapshot_id
5. In-flight Tasks: finish on intake snapshot OR policy-driven drain (product choice)
```

**Anti-pattern EVOL-AP-01:** Editing production YAML without version bump — breaks audit and replay.

### 49.2.4 Typed migration primitives (**Done** · APP-EVOL-2b)

`ApplicationMigration` orchestrates **typed** sub-migrations — one schema per primitive, composable in CI:

```text
ProfileMigration:
    migration_id: str
    from_spec_version: str
    to_spec_version: str
    field_transforms: list[FieldTransform]
    default_injection: dict              # new fields with safe defaults
    breaking: bool

GraphSpecMigration:
    migration_id: str
    from_graph_version: semver
    to_graph_version: semver
    node_renames: dict[str, str]
    edge_rewrites: list[EdgeRewrite]
    removed_nodes_policy: fail | orphan_audit

OrgEnvelopeMigration:
    migration_id: str
    from_envelope_version: semver
    to_envelope_version: semver
    playbook_id_map: dict[str, str]
    tool_deny_additions: list[str]
    tool_deny_removals: list[str]
```

**Rules:**

- Each primitive migration MUST have a **golden replay** or eval scenario when `breaking=true`.
- `ProfileMigration` runs before `GraphSpecMigration` before `OrgEnvelopeMigration` (dependency order).
- Partial migrations are forbidden in STRICT — all three digests must match target snapshot (§49.1.2).

**Status:** `ApplicationMigration` + typed sub-migrations **Done** (`APP-EVOL-2` · `APP-EVOL-2b` · `application_migration.py` · `check_application_migrations.py`).

---

## 49.3 Capability Governance

Tier-3 routes work via **capability tokens** on `Task` and `AgentBinding.capabilities[]` (§24.2, §37.4). At scale, capabilities need a **lifecycle** independent of agent class names.

### 49.3.1 Capability registry model (normative · **Done** APP-EVOL-3)

```text
CapabilityDescriptor:                    # UAEP §42.27 — harness-wide
    capability: str                       # e.g. research.pipeline
    version: semver
    agent_id: str
    contract_version: str
    deprecated: bool
    superseded_by: str | null

CapabilityAlias:                         # APP-EVOL-3 Done
    alias: str                            # research.pipeline (legacy)
    canonical: str                         # research.orchestrate
    sunset_at: datetime | null

CapabilityDeprecation:
    capability: str
    version: semver
    notice_ref: str
    migration_guide_ref: str
    block_routing_after: datetime
```

### 49.3.2 Tier-3 binding rules

| Rule | Enforcement |
|------|-------------|
| Manifest roster lists **canonical** capabilities only | `EnvironmentSkillToolConsistencyCheck` |
| Deprecated capability in STRICT | Nexus routing policy blocks or warns (V-REM-ALG.1) |
| Breaking capability change | Major semver bump; alias window ≥ 14 days |
| `research.pipeline` retired | Remove from `AgentBinding`; keep alias redirect in registry until sunset |

**Example:** `research.pipeline` superseded by `research.orchestrate` — Tier-3 manifest updates bindings; harness registry serves alias during migration window.

**Status:** `CapabilityDescriptor` + `CapabilityAlias` **Done** (`capability_alias.py` · `capability_alias_wiring.py` · intake middleware APP-EVOL-3); retired-agent routing filter **Done** (V-REM-ALG.1).

---

## 49.4 Agent Lifecycle Governance

Today: `Application → AgentBinding → Agent`. At 500 agents, **which agents may run in production** must be explicit.

### 49.4.1 AgentLifecycle states

**Code:** `intergrax/contracts/agent_lifecycle_state.py` · `AgentLifecycleState`

```text
experimental → development → candidate → staging → production → deprecated → retired
```

Each Tier-2 agent contract carries `lifecycle_state` (ACP). Tier-3 **`AgentBinding`** references agents that MUST satisfy host policy.

### 49.4.2 Governance policies (**Done** · APP-EVOL-4)

```text
AgentApprovalPolicy:
    allowed_states_for_strict: list[AgentLifecycleState]   # default: [production]
    allow_staging_in_balanced: bool

AgentPromotionPolicy:
    required_gates: list[str]              # e.g. ACP-PROD-1, eval golden id
    min_eval_pass_rate: float | null

AgentCertification:
    agent_id: str
    agent_version: semver
    certified_at: datetime
    certified_by: str
    evidence_refs: list[str]               # test run ids, ADR links
```

### 49.4.3 Tier-3 enforcement

| Posture | Rule |
|---------|------|
| **STRICT production** | `registry_assembly_resolver` rejects non-`PRODUCTION` agents unless explicit waiver in product ARCHITECTURE |
| **STAGING host** | `STAGING` + `PRODUCTION` allowed |
| **Lab** | All states except `RETIRED` (retired blocked — V-REM-ALG.1) |
| **Deprecation** | `evaluate_agent_lifecycle_transition()` — migration window + guide refs required |

**Promotion flow:** agent passes ACP-PROD gates → lifecycle `STAGING` → product host eval → `PRODUCTION` → added to `ApplicationManifest.agents`.

**Status:** lifecycle enum + transition evaluator **Done** (V-ALG.3); `AgentCertificationRecord` + STRICT roster gate **Done** (`agent_governance.py` · `agent_certification_wiring.py` · APP-EVOL-4).

---

## 49.5 Runtime Recovery

Reliability primitives exist (`ReliabilityProfile`, checkpoints, idempotency, compensation). Tier-3 needs an explicit **Application Recovery Contract** — what the **host** guarantees after failure.

### 49.5.1 Failure scenarios

| Scenario | Detection | Tier-3 host responsibility |
|----------|-----------|----------------------------|
| **Host process crash** | K8s / supervisor restart | Factory idempotent bootstrap; scheduler resumes pending tasks |
| **Container restart** | Lifespan hook | `wire_long_running_scheduler` + checkpoint store |
| **Partial graph execution** | Graph node failure | Nexus retry policy; `ApplicationRunSummary` partial status |
| **Node failure (single agent)** | Agent run FAILED | Orchestration retry / alternate binding (graph policy) |
| **HITL pause** | `HitlEscalationState` §42 | `mount_harness_task_routes` resume endpoint |
| **Budget hard exceed** | §43 | Terminal or HITL per `BudgetReactionProfile` |

### 49.5.2 Recovery actions (normative)

```text
ApplicationRecoveryContract:
    on_host_restart: resume_scheduler | cold_start_only
    on_task_interrupted: resume | restart | escalate_hitl
    on_graph_node_failure: retry_node | skip_with_audit | abort_graph
    on_corrupt_checkpoint: replay_from_snapshot | abort_with_incident
    max_resume_attempts: int
    preserve_snapshot_id: bool = true
```

| Action | When | Harness mechanism |
|--------|------|-------------------|
| **`resume`** | Checkpoint exists, same snapshot | `resume_token` + task checkpoint store |
| **`restart`** | Idempotent task, no partial side effects | New `task_id`, same payload + idempotency key |
| **`rollback`** | Mutating tool failure | Compensation queue (ACP-PROD-5) |
| **`replay`** | Lab / simulation | `EnvironmentSnapshot` + trace replay |

### 49.5.3 Product host checklist

Mutating STRICT hosts MUST document in product `ARCHITECTURE.md`:

1. Checkpoint store path and retention
2. Scheduler enabled for async/long-running
3. Recovery action per scenario above
4. Whether in-flight tasks drain on deploy or abort

**Cross-ref:** [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) §33.3, §34.4 · ACP checkpoint host wiring.

**Status:** typed `ApplicationRecoveryContract` on `ReliabilityProfile` **Done** (`application_recovery_contract.py` · `recovery_contract_wiring.py` · APP-EVOL-5).

---

## 49.6 Environment Diff and Audit

Large agent environments require **diff**, not eyeballing YAML.

### 49.6.1 ApplicationEnvironmentDiff (**Done** · APP-EVOL-6)

```text
ApplicationEnvironmentDiff:
    left_snapshot_id: str
    right_snapshot_id: str
    profile_diff: StructuredDiff
    graph_diff: StructuredDiff | null
    envelope_diff: StructuredDiff | null
    roster_diff: list[RosterEntryChange]
    risk_level: low | medium | high
    breaking_changes: list[str]
```

### 49.6.2 Diff operations

| Function | Input | Output |
|----------|-------|--------|
| `diff_profile(a, b)` | Two `ApplicationEnvironmentProfile` | Field-level changes, execution_mode delta |
| `diff_graph(a, b)` | Two `ApplicationGraphSpec` | Added/removed nodes, edge changes |
| `diff_envelope(a, b)` | Two `OrganizationalPolicyEnvelope` | Tool denies, playbook, channel changes |
| `diff_roster(a, b)` | Two `AgentBinding[]` | Capability/agent swaps |

### 49.6.3 Audit use cases

- **Pre-deploy review:** `diff(snapshot_prod, snapshot_candidate)` in CI
- **Incident:** compare `profile_snapshot_id` on failed Task vs current deploy
- **Org simulation:** diff envelope before enabling new playbook

**CLI:** `intergrax doctor diff-app --app legal --left 0.1.0 --right 0.2.0` (`doctor_diff_app.py` · `--json` · `--fail-on-high`).

**Status:** **Done** (`application_environment_diff.py` · `environment_diff_wiring.py` · `check_application_environment_diff.py` · APP-EVOL-6).

---

## 49.7 Application Packaging and Distribution

Intergrax composes **Applications + Agents + Skills + Tools + Profiles**. A formal **package** model enables marketplace-style distribution without forking the harness.

### 49.7.1 ApplicationPackage (**Done** · APP-EVOL-7)

```text
ApplicationPackage:
    package_id: str                        # e.g. com.intergrax.research
    app_id: str
    version: semver
    manifest: ApplicationManifest          # frozen
    dependencies: list[ApplicationDependency]
    distribution: ApplicationDistribution
```

```text
ApplicationDependency:
    kind: agent | skill | tool | integration | profile_fragment
    ref: str                               # slug or version pin
    version_constraint: str                # semver range
    optional: bool = false
```

```text
ApplicationDistribution:
    channel: local | git | registry | marketplace
    artifact_uri: str | null
    checksum: sha256
    signature_ref: str | null
```

### 49.7.2 Dependency closure

At `wire_application_environment()` time, resolver MUST verify:

```text
manifest.agents[]           → agent packages present in registry
environment tool/skill ids  → subset of catalogs (existing conformance)
integration_profile         → providers available
graph_spec nodes            → roster capabilities satisfied
```

**Scaffold today:** `new-stack` bundles agent + application; `agent_catalog.py` resolves specs — precursor to full `ApplicationPackage`.

### 49.7.3 Distribution rules

| Rule | Rationale |
|------|-----------|
| Package is **immutable** at a version | Reproducible deploys |
| Dependencies pinned in STRICT | No surprise catalog drift |
| Secrets never in package | `.env.example` only |
| Business logic stays Tier-2 | Package wires, does not embed cognition |

**Status:** **Done** (`application_package.py` · `package_wiring.py` · `package_emit.py` · `check_application_package.py` · APP-EVOL-7).

---

## 49.8 Implementation register (APP-EVOL)

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| APP-EVOL-1 | `EnvironmentSnapshot` + snapshot capture on intake | **Done** | `test_environment_snapshot_wiring.py` · ADR-APP-002 |
| APP-EVOL-2 | `ApplicationMigration` schema + validator CLI | **Done** | `check_application_migrations.py` |
| APP-EVOL-3 | `CapabilityAlias` registry + sunset routing | **Done** | `check_capability_alias_registry.py` |
| APP-EVOL-4 | `AgentCertification` + STRICT roster gate | **Done** | `check_agent_certification_roster.py` |
| APP-EVOL-5 | `ApplicationRecoveryContract` on `ReliabilityProfile` | **Done** | `check_application_recovery_contract.py` |
| APP-EVOL-6 | `ApplicationEnvironmentDiff` + `doctor diff-app` | **Done** | `check_application_environment_diff.py` |
| APP-EVOL-7 | `ApplicationPackage` + dependency resolver | **Done** | `check_application_package.py` · `package.json` from scaffold |

**Explicitly out of scope:** marketplace UI (H-APP deferred); Nexus fork; Tier-3 cognition loop.

---

# 50. Platform Operations Canon

Final freeze-ready layer for **reference platform architecture** — connects Tier-3 environments to harness-wide **capability graph**, **operational ownership**, **health scoring**, and **registry** surfaces. Does not alter Nexus, `ApplicationHost`, profile/graph/envelope primitives, or hook semantics (§32).

**Symmetry with ACP:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) describes the **executing unit** (agent); this document describes the **executing environment** (application). Together they form two peer pillars:

```text
Agent (ACP)          → how one unit thinks, acts, certifies, deprecates
Application (TIER3)  → how the environment composes, constrains, evolves, operates
```

---

## 50.1 Capability Graph (environment-scoped)

> **Canonical graph model:** ACP §19 — this section covers **Tier-3 environment view and ops** only; do not fork graph taxonomy here.

`ApplicationPackage.dependencies` (§49.7) lists **direct** refs. **CapabilityGraph** models the full **transitive** harness chain from IDEAL §19.4:

```text
Integration → Tool → Skill → Policy → Agent → Application → Product
```

### 50.1.1 Harness graph (Tier-0/1)

**Code:** `intergrax/runtime/architecture/capability_graph.py`

```text
CapabilityGraph:
    nodes: list[CapabilityNode]           # CapabilityNodeType enum
    edges: list[CapabilityEdge]           # DEPENDS_ON, CONSTRAINED_BY, SUPERSEDES, ...

CapabilityLineageReport:                 # V-CG.2
    records: upstream / downstream per node_id

CapabilityImpactReport:                  # V-CG.3
    impacts: blast_radius_node_ids per changed node
```

### 50.1.2 Environment graph view (Tier-3)

**Code:** `intergrax/applications/_shared/capability_graph_wiring.py` · `EnvironmentCapabilityGraphView`

Builds an **application-scoped subgraph** from:

- `ApplicationManifest` + `AgentBinding` roster
- `HarnessRegistrySnapshot` (tools, skills, prompts enabled by profile)
- Catalog graph seed via `build_catalog_capability_graph()`

```text
wire_environment_capability_graph(manifest, env, snapshot)
    → EnvironmentCapabilityGraphView
    → subset of global CapabilityGraph reachable from application node
```

### 50.1.3 Operations the graph enables

| Operation | API / report | Question answered |
|-----------|--------------|-------------------|
| **Lineage** | `build_capability_lineage_report(graph)` | What upstream integrations/tools feed this agent? |
| **Blast radius** | `build_capability_impact_report(graph)` | If tool X changes, what else breaks? |
| **Impact preview** | `policy_change_impact.py` | Policy deny addition — affected nodes |
| **Deprecation** | `SUPERSEDES` edge + §49.3 alias | Safe sunset window for `research.pipeline` |
| **Deploy review** | env graph diff vs previous snapshot | Unexpected new dependencies? |

### 50.1.4 Tier-3 rules

| Rule | Rationale |
|------|-----------|
| Every product host SHOULD expose graph view in ops/debug (read-only) | Impact analysis before profile edits |
| STRICT deploy CI SHOULD fail when blast radius includes uncertified agent | Governance §49.4 |
| Graph is **derived** from manifest + profile — not hand-edited parallel truth | Single source of composition |
| `ApplicationDependency` MUST resolve to graph node ids | Package ↔ graph linkage |

**Gap vs package-only model:** `ApplicationPackage` knows **what** depends on **what**; `CapabilityGraph` knows **impact**, **lineage**, and **blast radius** — required for platform-scale change management.

**Status:** harness graph + lineage/impact **Done** (V-CG.1–3); Tier-3 `EnvironmentCapabilityGraphView` **Done**; APP-OPS-1 STRICT deploy gate **Done** (`capability_graph_deploy_gate.py`).

---

## 50.2 Application ownership and operational responsibility

Agents have production ownership (V-ALG.4 · `production_ownership.py` · `OnCallOwnershipRegistry` for roster). **Applications** need the same operational contract at environment level.

### 50.2.1 ApplicationOperationalOwnership (**Done** · APP-OPS-2)

```text
ApplicationOperationalOwnership:
    app_id: str
    owner: ApplicationOwner              # business/accountable party
    maintainer: ApplicationMaintainer     # engineering team shipping host
    escalation: ApplicationEscalationContact
    on_call_rotation: str | null         # PagerDuty/Slack handle
    runbook_ref: str
    architecture_ref: str                # product ARCHITECTURE.md path
    status_page_component: str | null
```

```text
ApplicationOwner:
    name: str
    team: str
    contact: str

ApplicationMaintainer:
    team: str
    primary_contact: str
    repo_path: str                        # applications/<app>/

ApplicationEscalationContact:
    channel: slack | email | pagerduty | webhook
    target: str
    severity_routing: dict[str, str]      # sev1 → ..., sev3 → ...
```

### 50.2.2 Where it lives

| Surface | Field | Status |
|---------|-------|--------|
| `ApplicationManifest` | `ownership: ApplicationOperationalOwnership \| null` | **Done** APP-OPS-2 |
| Product `ARCHITECTURE.md` frontmatter | owner, maintainer, on-call | **Required today** (informal) |
| `ApplicationEnvironmentProfile` | inherit from manifest | **Deferred P4** — manifest gate sufficient today |
| APP-PROD gate | product hosts must declare ownership | **Done** `check_application_ownership.py` |

### 50.2.3 Enforcement

| Posture | Rule |
|---------|------|
| **PRODUCT profile** | `evaluate_application_ownership()` must pass before deploy tag |
| **Incident** | `ApplicationRunSummary` + ownership → escalation routing |
| **HITL / budget exceed** | `ApplicationEscalationContact` used by §43 notify reactions |

**Symmetric agent rule:** roster agents still require `ProductionOwnerMetadata` per contract (V-ALG.4); application ownership covers **host / environment**, not per-agent substitution.

**Status:** agent ownership **Done**; application ownership schema **Done** (`operational_ownership.py` · APP-OPS-2).

---

## 50.3 Architecture health model

APP-PROD and APP-EVOL gates are **boolean pass/fail**. At platform scale, operators need a **continuous health score** per application and per deployed environment.

### 50.3.1 EnvironmentHealthScore (**Done** · APP-OPS-3)

```text
EnvironmentHealthScore:
    app_id: str
    snapshot_id: str | null
    scored_at: datetime
    overall: float                         # 0.0 – 1.0
    dimensions: list[HealthDimensionScore]
    blockers: list[str]                    # hard failures
    warnings: list[str]
```

```text
HealthDimensionScore:
    dimension: HealthDimension
    score: float
    evidence_refs: list[str]
    stale_after: datetime | null
```

```text
HealthDimension (enum):
    deprecated_capabilities
    stale_agents                         # lifecycle < PRODUCTION in STRICT roster
    failed_migrations
    policy_coverage                      # org envelope eval golden pass rate
    test_coverage                        # §44 scenario matrix completeness
    ownership_complete
    capability_graph_valid
    budget_governance_configured
    recovery_contract_documented
```

### 50.3.2 ApplicationHealthScore

Rollup across **all registered environments** for one `app_id`:

```text
ApplicationHealthScore:
    app_id: str
    environments: list[EnvironmentHealthScore]
    worst_environment: str | null
    production_ready: bool                 # all prod envs ≥ threshold
```

### 50.3.3 Scoring rules (normative targets)

| Dimension | Green (≥0.9) | Red trigger |
|-----------|--------------|-------------|
| `deprecated_capabilities` | zero deprecated caps in roster | any deprecated cap in STRICT |
| `stale_agents` | all roster agents PRODUCTION | STAGING agent in prod host |
| `failed_migrations` | last migration CI green | breaking bump without migration |
| `policy_coverage` | UC-A7 golden pass | POLICY_DENIED on happy path |
| `test_coverage` | §44 rows pass for posture | missing scenario test |
| `capability_graph_valid` | no orphan nodes in env graph | unreachable agent node |

**CLI:** `intergrax doctor health-app --app legal` (`--json` · `--write` · `--fail-below`).

**Relation to §42:** `EnvironmentHealthStatus` on `ApplicationEnvironmentState` is **runtime task-scoped**; `EnvironmentHealthScore` is **ops platform-scoped** — complementary, not duplicate.

**Status:** **Done** (`environment_health_score.py` · `health_score_wiring.py` · `check_application_health_score.py` · APP-OPS-3).

---

## 50.4 Application and environment registry

Platform engineering surface — **inventory** of what exists, where it runs, at which version. Distinct from runtime Nexus registry (agent instances).

### 50.4.1 ApplicationRegistry **Done** (APP-OPS-4)

```text
ApplicationRegistry:
    entries: list[ApplicationRegistryEntry]

ApplicationRegistryEntry:
    app_id: str
    name: str
    current_version: semver
    package_ref: ApplicationPackage | null
    ownership: ApplicationOperationalOwnership
    health: ApplicationHealthScore | null
    registered_at: datetime
    source: git | manual | marketplace
```

**Operations:**

- `list_applications()` — all Tier-3 packages in monorepo + external
- `get_application(app_id)` — manifest + latest health
- `register_application(package)` — on scaffold / CI publish

### 50.4.2 EnvironmentRegistry **Done** (APP-OPS-4)

A **deployed instance** of an application (lab, staging, prod, tenant-specific):

```text
EnvironmentRegistry:
    entries: list[EnvironmentRegistryEntry]

EnvironmentRegistryEntry:
    environment_id: str                    # e.g. research-prod-eu1
    app_id: str
    app_version: semver
    profile_id: str
    execution_mode: ExecutionMode
    deployment: EnvironmentDeployment
    snapshot_id: str | null               # last known EnvironmentSnapshot
    health: EnvironmentHealthScore | null
```

```text
EnvironmentDeployment:
    channel: local | docker | k8s | serverless
    region: str | null
    image_tag: str | null
    endpoint: str | null
    deployed_at: datetime
    deployed_by: str
```

### 50.4.3 Registry operations

| Command | Returns |
|------------------|---------|
| `intergrax apps list` | All applications |
| `intergrax apps show <app_id>` | Versions, ownership, health |
| `intergrax envs list [--app <id>]` | All environments |
| `intergrax envs show <env_id>` | Deployment, snapshot, graph summary |

**Storage:** file-based registry in monorepo (`build/application_registry.json`, `build/environment_registry.json`).

**CLI:** `intergrax apps list|show|sync` · `intergrax envs list|show`.

**Status:** **Done** (`application_registry.py` · `registry_ops_wiring.py` · `check_application_registry.py` · APP-OPS-4). Ops automation should prefer registry artifacts over `applications/README.md`.

---

## 50.5 Implementation register (APP-OPS)

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| APP-OPS-1 | STRICT deploy gate: `EnvironmentCapabilityGraphView` + blast radius check | **Done** | `check_capability_graph_strict_deploy.py` |
| APP-OPS-2 | `ApplicationOperationalOwnership` on manifest + APP-PROD gate | **Done** | `check_application_ownership.py` |
| APP-OPS-3 | `EnvironmentHealthScore` + `doctor health-app` | **Done** | `check_application_health_score.py` |
| APP-OPS-4 | `ApplicationRegistry` + `EnvironmentRegistry` + CLI | **Done** | `check_application_registry.py` |
| APP-EVOL-2b | `ProfileMigration` / `GraphSpecMigration` / `OrgEnvelopeMigration` | **Done** | `migration_wiring.py` typed validators |

**Architecture freeze boundary:** APP-OPS-1..4 **Done** — Tier-3 canon is **feature-complete** for reference platform; remaining work is implementation, not structural redesign.

---

# 51. Cross-Document Consistency (Freeze)

Pre-freeze **semantic audit** — overlap between Tier-3, ACP, UAEP, and IDEAL. Full evidence: [`guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](../guides/GOVERNANCE_CONSISTENCY_AUDIT.md).

## 51.1 Verdict (2026-06-11)

| Question | Result |
|----------|--------|
| Two definitions of capability? | **No** — routing (`CapabilityDescriptor` / `AgentRegistry`) vs structure (`CapabilityGraph`) are layered |
| Two registries for the same thing? | **No** — runtime `AgentRegistry` ≠ ops `ApplicationRegistry` / `EnvironmentRegistry` |
| Ownership duplicates lifecycle? | **No** — lifecycle = state; ownership = on-call contacts (agent vs application scopes) |
| Health score duplicates APP-PROD? | **No** — gates = boolean blockers; score = continuous rollup |
| §50 vs IDEAL conflict? | **No** |

## 51.2 Naming risks (glossary discipline)

| Do not introduce | Use instead |
|------------------|-------------|
| `CapabilityRegistry` | `AgentRegistry` (routing) + `CapabilityGraph` (dependencies) |
| `GovernanceProfile` as ownership | `ApplicationOperationalOwnership` (§50.2) or `ProductionOwnerMetadata` (ACP §20) |
| `applications/README.md` as ops registry | `ApplicationRegistry` when APP-OPS-4 ships |

## 51.3 Canonical split (two pillars)

```text
ACP §12–§45   → executing unit (agent): contract, cognition, lifecycle, certification
TIER3 §24–§50 → executing environment (application): profile, hooks, evolution, ops
Shared        → CapabilityGraph (ACP §19), routing (UAEP §42.27), registries (ACP §18)
```

**Freeze status:** Tier-3 structural architecture **approved** with glossary rules above.

---

**Plan:** [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](../plan/TIER3_APPLICATION_ENVIRONMENT.md) — [fidelity matrix](../plan/TIER3_APPLICATION_ENVIRONMENT.md#architecture-fidelity-matrix--20-51) · [APP-* master backlog](../plan/TIER3_APPLICATION_ENVIRONMENT.md#master-implementation-backlog-app-unified) · phases H-APP-CON · H-APP-EVOL · H-APP-OPS · H-APP-FREEZE  
**Consistency audit:** [`guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](../guides/GOVERNANCE_CONSISTENCY_AUDIT.md)

---
