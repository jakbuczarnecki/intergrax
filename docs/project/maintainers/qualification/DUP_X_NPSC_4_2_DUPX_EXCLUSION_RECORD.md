# DUP-X — NPSC-4.2 exclusion record (orchestration backend allowlist)

| Field | Value |
| --- | --- |
| **Gate** | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` |
| **Failing test @ HEAD** | `test_npsc42_orchestration_backend_access_confined_to_allowlist` |
| **Allowlist SSOT in test** | `_ALLOWED_ORCHESTRATION_BACKEND_ACCESSORS` (4 paths) |
| **Reconciliation HEAD** | `0159c2a18b39547bcd60823489c1a4bea1229b38` |
| **Replay log** | `.tmp/proof/dup-x-03-final-evidence/npsc-run.log` |
| **Wave 1 action** | Document only — **no** NPSC allowlist edit |

## Violating modules (@ reconciliation HEAD)

Production references to `_orchestration_backend` outside the frozen allowlist:

| # | Module path |
| --- | --- |
| 1 | `intergrax/applications/_shared/harness_host_runtime.py` |
| 2 | `intergrax/applications/_shared/scenario_runtime_baseline.py` |
| 3 | `intergrax/runtime/execution/application_host_orchestration_composition.py` |
| 4 | `intergrax/runtime/execution/environment_host_task_execution.py` |
| 5 | `intergrax/runtime/execution/environment_orchestration_materialization.py` |
| 6 | `intergrax/runtime/execution/harness_host_internal_composition.py` |
| 7 | `intergrax/runtime/execution/harness_host_orchestration_wiring.py` |
| 8 | `intergrax/runtime/execution/harness_host_topology_submission_wiring.py` |
| 9 | `intergrax/runtime/execution/host_orchestration_assembly_validation.py` |
| 10 | `intergrax/runtime/execution/worker_host_task_execution_composition.py` |

**Suite @ HEAD:** 26 passed · 1 failed (above) · other NPSC-4.2 tests **PASS** (including raw NexusLoop composition-owner allowlist).

## DUP-X class mapping (DUP-08 · DUP-14)

| DUP class | DUP-X obligation | Relation to NPSC failure |
| --- | --- | --- |
| **DUP-08** | Bypass / lab import fence — **CAND-15 RATIFIED** | L2: `test_dup_x_compat_r3_lab_paths_not_imported_from_sanctioned_roots` — **not** orchestration-backend token scan |
| **DUP-14** | CONFIG-X ∪ composition hints SSOT — **PASS** | L2: `test_dup_x_exit_composition_hints_normative_in_config_x` — **not** harness `_orchestration_backend` accessor policy |

NPSC drift is **architecture gate backlog** (stale `_ALLOWED_ORCHESTRATION_BACKEND_ACCESSORS` vs expanded canonical harness/runtime composition). It does **not** reopen DUP-08 bypass census or DUP-14 CONFIG-X reconciliation.

## Verdict (DUP-X mandatory scope)

| Decision | Rationale |
| --- | --- |
| **Exclude from DUP-X mandatory L2** | Primary qual SSOT remains `tests/qualification/dup_x/` (36 passed @ reconciliation HEAD) |
| **Not a DUP-X BLOCKER** | Parent closure mechanical gates **PASS**; COMPAT-R3 + CONFIG-X exit gates **PASS** |
| **Track elsewhere** | NPSC-4.2 / EBH harness composition program — operator may refresh allowlist outside DUP-X |

Cross-ref: [`DUP_X_FINDINGS_REGISTER.md`](DUP_X_FINDINGS_REGISTER.md) § NPSC-4.2 exclusion.

## L3 narrative — representative modules (composition wiring only)

Auditor spot-check: these modules reference `_orchestration_backend` for **sanctioned harness/EE composition wiring** — not Integration slug selection, not Governance bypass, not alternate execution ingress.

| Module | Role | DUP-X relevance |
| --- | --- | --- |
| `harness_host_orchestration_wiring.py` | Wires orchestration backend into harness host graph via `_orchestration_backend_access` helpers | Composition bind only — TRACE-X sanctioned harness path |
| `application_host_orchestration_composition.py` | Assembles application-hosted orchestration materialization | Delegates to platform execution composition — no shadow registry |
| `harness_host_runtime.py` | Tier-3 shared harness host runtime surface | Uses backend accessor for hosted profile wiring — admission remains Governance-attached upstream |

**Caller invariant:** provider/category selection remains `intergrax/integrations/registry/factory.py` + `_shared/config.py` (Class D matrix). **Independent B6:** auditor must accept or reject this exclusion explicitly (see Protocol 03 PASS record template).

