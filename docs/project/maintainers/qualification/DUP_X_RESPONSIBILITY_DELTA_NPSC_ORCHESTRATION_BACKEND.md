# DUP-X — Responsibility Delta: NPSC-4.2 orchestration backend (B4 boundary)

| Field | Value |
| --- | --- |
| **Scope** | Ten production modules @ [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md) |
| **DUP classes** | DUP-08 (bypass fence) · DUP-14 (composition SSOT) |
| **Canonical execution owner** | `intergrax/runtime/execution/environment_orchestration_materialization.py` — `materialize_host_orchestration_backend` |
| **Canonical accessor** | `intergrax/runtime/execution/_orchestration_backend_access.py` — `orchestration_backend_for_execution_engine` (EE owner zone only) |
| **Integration / provider owner** | `intergrax/integrations/registry/factory.py` — `resolve_from_profile` (unchanged; no module in ten-set reads env slug) |
| **SYS-INV linkage** | **SYS-INV-01** · **SYS-INV-02** (tier import boundaries) · **FRZ-OWN-01** (exactly-one semantic owner — guard matrix / EBH OWN bundle; not a `SYS-INV-*` row in [`SYSTEM_INVARIANTS.md`](../../technical/guides/SYSTEM_INVARIANTS.md)) — execution ingress owned by EE materialization + admitted host task ports, not duplicate Integration selection |

## Ten-module classification matrix

| # | Module | `_orchestration_backend` use | Immediate production caller(s) | Pass vs materialize | Independent execution init? | Governance bypass? | Authority in control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `harness_host_runtime.py` | Calls `materialize_host_orchestration_backend`; wiring via `orchestration_application_wiring_target_from_materialization` | Tier-3 harness host factories (`applications/*/…/host_factory.py` paths) | **Materializes** inside sanctioned harness composition root | **No** — wires `HostOrchestrationApplicationWiringTarget` / assembly validators; run path uses host execution surface | **No** — security/guardrail assembly + observability on materialized host | **HarnessHostRuntime** composition root · EE materialization owner |
| 2 | `scenario_runtime_baseline.py` | Import `host_orchestration_backend_spec_builder` (token match only) | Scenario platform bootstrap in same module | **Neither** — delegates to `compose_application_host_orchestration_session` | **No** — consumes `ApplicationHostOrchestrationSession.host_execution` | **No** — `build_harness_host_task_execution_governance()` supplies `root_authority_admission` + `admit_root_governance_identity` | **EE** `compose_application_host_orchestration_session` + governance bundle |
| 3 | `application_host_orchestration_composition.py` | `materialize_host_orchestration_backend` + `orchestration_backend_for_execution_engine` | `scenario_runtime_baseline.py` · application host orchestration entry | **Both** — single EE composition step | **No** — returns session + materialization for EE-internal wiring only | **No** — requires `RootExecutionAuthorityAdmissionPort` + revision admission in `build_environment_host_task_execution` | **Execution Engine** composition owner |
| 4 | `environment_host_task_execution.py` | `orchestration_backend_for_execution_engine(materialization)` → `build_host_task_execution` | `application_host_orchestration_composition.py` | **Pass** approved materialization | **No** — host task port only | **No** — admission ports mandatory parameters | **`nexus_host_execution.build_host_task_execution`** + governance ports |
| 5 | `environment_orchestration_materialization.py` | Defines `materialize_host_orchestration_backend` / `_backend` handle | EE composition modules (#3, #7, #10) + #1 | **Materializes** Nexus inside owner zone | **No** — construction only; run requires host task wiring | **No** — spec carries `governance_evidence_recorder`, `authority_policy`, etc. | **Execution Engine** — sole NexusLoop constructor for host orchestration |
| 6 | `harness_host_internal_composition.py` | Accessor unwrap for harness capabilities | `harness_host_runtime.py` @ `_internal_composition=` | **Pass** | **No** — exposes terminals/bus/plugins only | **No** | **HarnessHostInternalComposition** (private to runtime) |
| 7 | `harness_host_orchestration_wiring.py` | Accessor → wiring target protocols | `harness_host_runtime.py` (application wiring targets) | **Pass** | **No** | **No** | Contract-typed wiring targets (not Integration) |
| 8 | `harness_host_topology_submission_wiring.py` | Accessor → strict topology submission port | Harness topology wiring from host runtime | **Pass** | **No** | **No** — uses `MeaningfulSideEffectAuthorizationPort` | **Topology production composition** owner |
| 9 | `host_orchestration_assembly_validation.py` | Accessor for assembly/security/guardrail/reliability wiring | `scenario_runtime_baseline.py` · post-materialization validation | **Pass** | **No** | **No** | Tier-3 assembly validators (read-only checks + subscriptions) |
| 10 | `worker_host_task_execution_composition.py` | Materialize + accessor → `build_host_task_execution` | Worker / long-running host composition (runtime/task) | **Both** | **No** | **No** — `root_authority_admission` required | **Worker host execution** port under EE |

## Positive evidence (code-grounded)

- **Single materialization owner:** production `materialize_host_orchestration_backend(` call sites are exactly four paths (definition + three composers + harness root) — gate `test_b4_materialize_host_orchestration_backend_callers_allowlisted`.
- **Accessor owner zone:** `orchestration_backend_for_execution_engine` imports exist only under `intergrax/runtime/execution/**` — gate `test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone`.
- **Governance on run path:** scenario and worker composers require `RootExecutionAuthorityAdmissionPort` and `admit_root_governance_identity` — gate `test_b4_ten_modules_execution_wiring_requires_governance_ports_in_call_chain`.
- **COMPAT-R3 / Class D:** ten modules contain no lab bypass markers; none appear in shadow env-slug census — gate `test_b4_ten_modules_exclude_lab_bypass_and_shadow_slug_selection` (same markers as Wave B).
- **Tier-3 Nexus import fence:** no `from intergrax.runtime.nexus.nexus_loop import NexusLoop` under `intergrax/applications/**` (NPSC-4.2 separate gate still enforces composition-owner allowlist for shared wiring).

## Negative evidence (what would be a real B4 FAIL)

| Prohibited behavior | Detection | Observed @ qual baseline |
| --- | --- | --- |
| Second Nexus materialization outside allowlist | `test_b4_materialize_host_orchestration_backend_callers_allowlisted` | **None** |
| Tier-3 direct EE backend accessor import | `test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone` | **None** |
| Lab / transport bypass in ten-module surface | COMPAT-R3 markers scan | **None** |
| Shadow Integration slug selection | `discover_integration_env_slug_reads_outside_factory` | **None** in ten-set |
| Naked orchestration run without admission ports | governance port assertions | **Not present** on scenario/worker/app composition chain |

## NPSC-4.2 gate cross-check (unchanged)

| Artifact | Relationship |
| --- | --- |
| `test_npsc42_orchestration_backend_access_confined_to_allowlist` | **Token-presence** scan vs **stale 4-path allowlist** — still **FAIL** @ baseline (expected) |
| DUP-X B4 negative gates | **Behavioral** boundary — **PASS** @ baseline |
| Sanctioned composition roots | `discover_sanctioned_production_composition_roots()` — harness/scenario paths compose via EE/session, not parallel Integration owner |

## B4 auditor recommendation

| Question | Answer |
| --- | --- |
| Second Execution authority? | **No** — one materialization function; accessors unwrap existing materialization only |
| Governance bypass? | **No** — admission ports required on host task execution paths |
| Parallel composition ownership? | **No** — CONFIG-X hints + sanctioned roots; modules bind wiring to already-materialized backend |
| NPSC exclusion (Outcome A+B)? | **Recommend ACCEPT for DUP-X B4** — document NPSC allowlist refresh as **separate NPSC backlog**, not DUP-X BLOCKER |
