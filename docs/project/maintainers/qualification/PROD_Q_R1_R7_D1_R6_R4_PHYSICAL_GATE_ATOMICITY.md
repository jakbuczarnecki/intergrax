# PROD-Q-R1-R7-D1-R6-R4 — Physical Gate Atomicity & Final Concurrency Qualification

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (implementation evidence — not qualification closure) |
| **Parent** | **PROD-Q-R1-R7-D1-R6** — **REJECTED** / corrective chain |
| **ADR** | **ADR-01-R6** — **DEC-09** / **DEC-09-EXT** |

## Remediation summary (R4 root causes)

| ID | Issue | Remediation |
|----|-------|-------------|
| R4-01 | Non-atomic unit-open mutations | Adapter-owned **stage → publish** under `RLock`; publish applies authorized flag, gate, lifecycle atomically with rollback |
| R4-02 | Late `physical_opened` compensation flag | Coordinator tracks `physical_admission_affected` after stage; SQLite **publish only after commit** |
| R4-03 | Independent port invocation | Staged apply / forged evidence cannot admit tasks without coordinator **publish** after canonical verification |
| R4-04 | Missing SQLite S2 | `test_sqlite_s2_admission_wins_concurrent` + in-memory S2 regression in R4 suite |
| R4-05 | Commit / lifecycle gaps | SQLite commit-failure after stage/reserve; drain clears staged; stop invalidation tests |

## Key modules

- `intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py` — stage/publish/compensate
- `intergrax/agent_distribution/activation_domain_coordinator.py` — publish-after-commit (SQLite), publish-in-work (in-memory)
- `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r4_physical_gate_atomicity.py`

## Explicit non-claims

- Does **not** close **D1-R6**, **D1/D1b**, **R7**, **R1**, or **PROD-Q**
- Does **not** promote any **FRZ-PRD** row to PASS
- Full **18-case** matrix and **S3 rollback orchestration** remain partially evidenced — see test module gaps in audit report
