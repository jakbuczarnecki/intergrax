# ROADMAP-REPLAY-X — NPSC-4.2 replay disposition

| Field | Value |
| --- | --- |
| **Gate** | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` |
| **Global allowlist** | **FAIL** @ HEAD (10 modules) |
| **DUP-X exclusion** | ACCEPT for DUP-X scope only — does **not** auto-exclude REPLAY-X |

## Per-module disposition (@ AUDITED_HEAD)

| # | Module | REPLAY disposition |
| --- | --- | --- |
| 1 | `intergrax/applications/_shared/harness_host_runtime.py` | **PASS (SCOPED)** — L2 scan; no replay invariant breach vs DUP-X B4 record |
| 2 | `intergrax/applications/_shared/scenario_runtime_baseline.py` | **PASS (SCOPED)** |
| 3 | `intergrax/runtime/execution/application_host_orchestration_composition.py` | **PASS (SCOPED)** |
| 4 | `intergrax/runtime/execution/environment_host_task_execution.py` | **PASS (SCOPED)** |
| 5 | `intergrax/runtime/execution/environment_orchestration_materialization.py` | **PASS (SCOPED)** |
| 6 | `intergrax/runtime/execution/harness_host_internal_composition.py` | **PASS (SCOPED)** |
| 7 | `intergrax/runtime/execution/harness_host_orchestration_wiring.py` | **PASS (SCOPED)** |
| 8 | `intergrax/runtime/execution/harness_host_topology_submission_wiring.py` | **PASS (SCOPED)** |
| 9 | `intergrax/runtime/execution/host_orchestration_assembly_validation.py` | **PASS (SCOPED)** |
| 10 | `intergrax/runtime/execution/worker_host_task_execution_composition.py` | **PASS (SCOPED)** |

**Meta:** Global allowlist remains **FAIL** until NPSC owner refreshes allowlist (Class B backlog). **vendor_neutrality_replay_failures** = **0** when dispositions hold and no new PROHIBITED-* introduced.

Cross-ref: [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md).
