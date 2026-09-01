# TIER3_APPLICATION_ENVIRONMENT - embedded detail

**Parent hub:** [`TIER3_APPLICATION_ENVIRONMENT.md`](../TIER3_APPLICATION_ENVIRONMENT.md)

## Architecture fidelity matrix - §20–§51

Maps each architecture section to **plan phase**, **implementation status**, **code anchor**, and **acceptance test**. **Done** = architecture row is implemented and gated unless marked *doc-only*.

| Arch § | Topic | Plan IDs | Status | Code / test anchor |
|--------|-------|----------|--------|-------------------|
| §20 | Shadow workspace lifecycle | H-APP.3.4 · APP-CON-8 · APP-PROD-8 | **Done** | `shadow_wiring.py` · `workspace_cleanup_wiring.py` · `check_workspace_cleanup.py` |
| §21 | Sandbox lifecycle | H-APP.3.5 · APP-CON-8 · APP-PROD-8 | **Done** | `sandbox_wiring.py` · `workspace_cleanup_wiring.py` · `test_workspace_cleanup_wiring.py` |
| §22 | `ApplicationEnvironmentProfile` | H-APP.1.* · **APP-EVOL-8** | **Done** - flat §22.1 + bundles §22.6 M1–M3 | `environment_profile` · `test_environment_profile_bundles.py` · ADR-APP-003 |
| §23 | Interaction postures | H-APP-DOC.* · H-APP-WIRING.* | **Done** | §23.7 matrix closed on reference hosts |
| §24 | `ApplicationManifest` / `AgentBinding` | N.1 · H-APP.1.2 | **Done** | `manifest.py` · `test_manifest_conformance.py` |
| §25 | `run_task` / `HarnessApplication` / `ApplicationHost` | APP-CON-1 · N.* | **Done** | `harness/app.py` · `test_application_host_wiring.py` |
| §26 | `ApplicationRunSummary` (Plane A) | ACP-OBS-2 · APP-CON-6 | **Done** | `application_run_summary_builder.py` · `run_artifact_bundle_builder.py` |
| §27 | Roster / registry assembly | N.2.1 · H-APP.1.4 | **Done** | `registry_assembly_resolver.py` |
| §28 | APP invariants (no app cognition loop) | H-APP-CON-DOC.* | **Done** | *doc-only* · rejected `on_next_orchestration_step` |
| §29–§31 | Terminology · control modes · facade | H-APP.0.* · APP-CON-DX.1 | **Done** | `APPLICATION_CREATION_GUIDE.md` |
| §32 | `ApplicationHost` hook surface | APP-CON-1 | **Done** | `application_host.py` · `hooks.py` |
| §32.6 | Hook ordering · conflicts · determinism | APP-CON-5 | **Done** | `hook_runtime_guard.py` · `middleware_hook_timeout_seconds` |
| §33 | Dual observability planes | ACP-OBS-* · H-APP.4.8 | **Done** | `test_application_run_summary_builder.py` |
| §34 | Per-agent `AgentBinding` / budget slice | H-APP.* · ACP §30 | **Done** | `merge_environment` · ACP plan Wave 2 |
| §35 | Use-case catalog UC-A* | APP-CON-7 | **Done** | `tier3_scenario_matrix_wiring.py` · `test_tier3_scenario_matrix.py` |
| §36 | Architecture synthesis | H-APP-CON-DOC.* | **Done** | *doc-only* |
| §37 | Pre-implementation APP-CON contracts | H-APP-CON-DOC.* | **Done** | *doc-only* |
| §38 | L4 execution stack | H-APP.3.3 · H-APP-WIRING | **Done** | `nexus_factory.py` · `build_harness_host_runtime` |
| §39 | `OrganizationalPolicyEnvelope` | ACP-ORG-* | **Done** | `org_policy.py` · `test_uc11_product_host_compliance.py` |
| §40 | APP-PROD gates | APP-PROD-1..9 | **Done** | `check_application_production_gates.py` · APP-PROD-6 `check_environment_state_usage.py` |
| §41 | Composition primitive separation | H-APP-CON-DOC.* | **Done** | *doc-only* |
| §42 | `ApplicationEnvironmentState` v2 | APP-CON-2 · APP-CON-3 | **Done** | `environment_state.py` · lifecycle middleware |
| §43 | Budget / token governance | ACP-TOK-* · APP-CON-3 · APP-PROD-7 | **Done** | see [Cross-plan §43](.#cross-plan--43-budget--token-governance) |
| §44 | Scenario test matrix | APP-CON-7 | **Done** | `check_tier3_scenario_matrix.py` · `-m tier3_scenario` |
| §45 | New application checklist | APP-CON-DX.1 · N.* | **Done** | `APPLICATION_CREATION_GUIDE.md` §3 |
| §46 | Production readiness criteria | APP-PROD-* · ACP-PROD-* · ACP-TOK-* | **Done** | APP-PROD-* **Done** · ACP-TOK-1..3 · ACP-TOK-CI **Done** |
| §47 | Developer mental model | APP-CON-DX.1 | **Done** | `APPLICATION_CREATION_GUIDE.md` §1 |
| §48 | Application artifacts | APP-CON-4 · APP-CON-6 | **Done** | `application_artifacts.py` · `run_artifact_bundle.v1` on summary |
| §49 | Runtime evolution | APP-EVOL-1..7 · APP-EVOL-2b | **Done** | `environment_diff_wiring.py` · `package_wiring.py` · §49.8 register |
| §50 | Platform operations | APP-OPS-1..4 | **Done** | `health_score_wiring.py` · `registry_ops_wiring.py` · `intergrax apps/envs` |
| §51 | Cross-doc consistency | H-APP-FREEZE-* | **Done** | `GOVERNANCE_CONSISTENCY_AUDIT.md` |

---

## Master implementation backlog (APP-* unified)

Single register for all open architecture rows. **Execution order:** [§6.2y](.#62y-phase-app-backlog-execution-order-post-freeze).

### APP-CON - host contracts (architecture §25–§32 · §42 · §48)

| ID | Arch § | Deliverable | Status | Acceptance |
|----|--------|-------------|--------|------------|
| APP-CON-1 | §25 · §32 | `ApplicationHost` in `build_harness_host_runtime` | **Done** | `test_application_host_wiring.py` |
| APP-CON-2 | §42 | `ApplicationEnvironmentState` v2 | **Done** | `test_environment_state_and_artifacts.py` |
| APP-CON-3 | §42 · §43 | Nexus lifecycle updates `app_env_state.v1` (phase, budget, HITL) | **Done** | `test_application_environment_state_lifecycle.py` |
| APP-CON-4 | §48 | Artifact ref models | **Done** | `application_artifacts.py` |
| APP-CON-5 | §32.6 | Hook timeout · error→BLOCK · audit events | **Done** | `test_hook_runtime_guard.py` · product 250ms timeout |
| APP-CON-6 | §26 · §48 | `RunArtifactBundle` on `ApplicationRunSummary.metadata` | **Done** | `test_task_finisher_artifact_bundle.py` |
| APP-CON-7 | §35 · §44 | Scenario matrix gate - UC-A* minimum per posture | **Done** | `tier3_scenario_matrix_wiring.py` · `check_tier3_scenario_matrix.py` · `-m tier3_scenario` |
| APP-CON-8 | §20–§21 | Shadow/sandbox refs in env state + lifespan cleanup | **Done** | `workspace_cleanup_wiring.py` · `test_workspace_cleanup_wiring.py` |
| APP-CON-DX.1 | §31 · §45 · §47 | Author guide APP appendix (mental model + checklist) | **Done** | `APPLICATION_CREATION_GUIDE.md` |
| APP-CON-DX.2 | §37 | Regenerate domain audit prompt for §24–§51 | **Done** | `check_tier3_audit_prompt.py` |

### APP-PROD - release gates (architecture §40 · §46)

| ID | Arch § | Deliverable | Status | Acceptance |
|----|--------|-------------|--------|------------|
| APP-PROD-1 | §40.2 | `check_application_production_gates.py` | **Done** | script exits 0 |
| APP-PROD-2 | §40.2 | Reference hosts use `build_harness_host_runtime` | **Done** | H-APP-WIRING |
| APP-PROD-3 | §40.2 | `ApplicationHost` mounted when provided | **Done** | `test_application_host_wiring.py` |
| APP-PROD-4 | §40.2 | Manifest conformance | **Done** | `test_manifest_conformance.py` |
| APP-PROD-5 | §40.2 | Deploy triad | **Done** | `test_application_deploy_triad.py` |
| APP-PROD-6 | §40.2 | `check_environment_state_usage` lint | **Done** | `environment_state_usage_wiring.py` · `check_environment_state_usage.py` |
| APP-PROD-7 | §40.2 · §43 | `check_budget_enforcement` on STRICT product hosts | **Done** | `check_budget_enforcement.py` · product manifests `budget_slice` |
| APP-PROD-8 | §20–§21 | `check_workspace_cleanup` lifespan hooks | **Done** | `check_workspace_cleanup.py` · `test_check_workspace_cleanup.py` |
| APP-PROD-9 | §40.2 | Wire APP-PROD-1 into `pytest -m gate` / CI | **Done** | `test_check_application_production_gates.py` · CI `gate-governance-tier` |

### APP-EVOL - evolution (architecture §49)

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| APP-EVOL-1 | `EnvironmentSnapshot` + intake `profile_snapshot_id` | **Done** | `test_environment_snapshot_wiring.py` · ADR-APP-002 |
| APP-EVOL-2 | `ApplicationMigration` schema + CI validator | **Done** | `application_migration.py` · `check_application_migrations.py` |
| APP-EVOL-2b | `ProfileMigration` / `GraphSpecMigration` / `OrgEnvelopeMigration` | **Done** | `migration_wiring.py` typed validators per §49.2.4 |
| APP-EVOL-3 | `CapabilityAlias` + deprecation routing | **Done** | `capability_alias_wiring.py` · `check_capability_alias_registry.py` |
| APP-EVOL-4 | `AgentCertification` + STRICT roster gate | **Done** | `agent_certification_wiring.py` · `check_agent_certification_roster.py` |
| APP-EVOL-5 | `ApplicationRecoveryContract` on `ReliabilityProfile` | **Done** | `application_recovery_contract.py` · `check_application_recovery_contract.py` |
| APP-EVOL-6 | `ApplicationEnvironmentDiff` + `doctor diff-app` | **Done** | `check_application_environment_diff.py` |
| APP-EVOL-7 | `ApplicationPackage` + dependency resolver | **Done** | `check_application_package.py` |
| APP-EVOL-8 | Hierarchical profile bundles (P1-ARCH-01 · §22.6) | **Done** - M1–M3 | ADR-APP-003 · phases M1–M3 below |

### APP-EVOL-8 - Hierarchical profile bundles (P1-ARCH-01)

**Status:** **Done** (2026-06-18) - architecture §22.6 + ADR-APP-003 **accepted**; **M1–M3 implemented** (nested bundles, flat shims, digest parity, schema gate, `spec_version` 2.0 wire + migration tooling)  
**Goal:** Reduce flat `ApplicationEnvironmentProfile` namespace growth (43+ top-level fields) by nesting existing sub-profiles into seven bundles **without** changing `APP-INV-06`, §41 primitives, or Nexus wiring in M1–M2.

**ADR:** [`ADR-APP-003`](../adr/entries/2026-06-17/ADR-APP-003.md)

| ID | Phase | Deliverable | Status | Priority | Acceptance |
|----|-------|-------------|--------|----------|------------|
| APP-EVOL-8-DOC | - | Architecture §22.6 + plan register + cross-doc sync | **Done** | **Critical** | This plan row · `architecture/TIER3_APPLICATION_ENVIRONMENT.md` §22.6 |
| APP-EVOL-8.1 | M1 | Bundle models (`HostMeta`, `SecurityEnvelope`, `CapabilityBundle`, `CognitionBundle`, `GovernanceBundle`, `TopologyBundle`, `IsolationBundle`, `EnvironmentExtensions`) nested on root | **Done** | `environment_profile/bundles.py` · `extra=forbid` on all bundles |
| APP-EVOL-8.2 | M1 | Flat `@property` shims (`env.tool_profile` → `env.capabilities.tools`) - zero wiring diff | **Done** | `environment_profile/root.py` · existing `*_wiring.py` tests green |
| APP-EVOL-8.3 | M1 | Flat JSON deserializer + bundle-normalized snapshot/diff digest parity | **Done** | `test_environment_profile_bundles.py` · `normalization.py` · `environment_snapshot_wiring.py` |
| APP-EVOL-8.4 | M2 | Per-bundle presets (`CapabilityBundle.lab()`, `GovernanceBundle.product()`, …) | **Done** | `bundles.py` · `lab_defaults()` / `product_defaults()` built from bundles |
| APP-EVOL-8.5 | M2 | Shared capability packs - reusable `CapabilityBundle` across manifests | **Done** | `reference_capability_bundle.py` · reference hosts import shared pack |
| APP-EVOL-8.6 | M3 | `spec_version: "2.0.0"` - nested JSON canonical; flat top-level deprecated | **Done** | Migration guide · `ProfileMigration` validator extension |
| APP-EVOL-8.7 | M1 | Gate: `check_environment_profile_bundle_schema.py` - export schema includes bundles | **Done** | `scripts/maintenance/check_environment_profile_bundle_schema.py` |

**Explicitly out of scope:** second composition root; Nexus profile fork; moving `AgentBinding` into bundles; marketplace UI.

**Suggested PR order:** APP-EVOL-8-DOC → APP-EVOL-8.1 → APP-EVOL-8.2 → APP-EVOL-8.3 → APP-EVOL-8.7 → APP-EVOL-8.4 → APP-EVOL-8.5 → APP-EVOL-8.6.

#### APP-EVOL-8.6 migration guide (1.x flat → 2.0 nested)

1. **Authoring (greenfield):** construct via nested bundles (`HostMeta`, `SecurityEnvelope`, …) and set `meta.spec_version="2.0.0"`, or call `profile.with_spec_v2_wire()` on an existing 1.x profile.
2. **Wire JSON:** `model_dump()` emits nested bundle roots only when `spec_version` starts with `2.`; 1.x remains flat for backward compatibility.
3. **Declarative migration:** register `ProfileMigration` with `from_spec_version` `1.0.0`, `to_spec_version` `2.0.0`, `breaking=true`, and `field_transforms` (or use `standard_profile_spec_v2_migration()`).
4. **Runtime apply:** `apply_profile_migration(profile, migration)` lifts flat JSON, bumps `meta.spec_version`, and validates nested canonical wire.
5. **Digest parity:** `bundle_normalized_payload()` / snapshot digests remain stable for semantically equal profiles; `spec_version` bump is intentional wire metadata.
6. **STRICT hosts:** adopt 2.0 only after golden replay / scenario matrix per §44 - reference hosts may remain on 1.x until product cutover.

### APP-OPS - platform operations (architecture §50)

| ID | Deliverable | Status | Acceptance |
|----|-------------|--------|------------|
| APP-OPS-1 | Env capability graph + blast radius STRICT gate | **Done** | `check_capability_graph_strict_deploy.py` · `test_capability_graph_deploy_gate.py` |
| APP-OPS-2 | `ApplicationOperationalOwnership` on manifest | **Done** | `check_application_ownership.py` · `test_operational_ownership_gate.py` |
| APP-OPS-3 | `EnvironmentHealthScore` + `doctor health-app` | **Done** | `check_application_health_score.py` |
| APP-OPS-4 | `ApplicationRegistry` + `EnvironmentRegistry` + CLI | **Done** | `check_application_registry.py` |

---

## Cross-plan - §43 budget / token governance

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

**Fidelity rule:** §43 **Done** - ACP-TOK-* complete, APP-CON-3 seeds `ActiveBudgetState`, APP-PROD-7 gates STRICT product manifests.

---

## Fidelity verification gates

Run after any Tier-3 PR touching hosts, contracts, or wiring:

```bash
# Tier-3 unit + host contracts
uv run pytest tests/unit/applications/ -q

# APP-PROD-1 (wire to gate via APP-PROD-9)
python scripts/gates/check_application_production_gates.py

# APP-PROD-6 typed env state on hooks
python scripts/maintenance/check_environment_state_usage.py

# APP-EVOL-8.7 bundle schema (M1)
python scripts/maintenance/check_environment_profile_bundle_schema.py

# Harness tier boundaries
python scripts/maintenance/check_harness_no_getattr.py
python scripts/maintenance/check_agent_registry_bypass.py

# Domain pair
python scripts/docs/check_docs_domain_pairs.py

# Full gate (includes agent + platform)
uv run pytest -m gate -q
```

**Architecture-complete Tier-3 DoD (target):** all rows in [Master backlog](.#master-implementation-backlog-app-unified) **Done** · fidelity matrix all **Done** · `GOVERNANCE_CONSISTENCY_AUDIT.md` glossary respected · no §51 naming violations.

---
