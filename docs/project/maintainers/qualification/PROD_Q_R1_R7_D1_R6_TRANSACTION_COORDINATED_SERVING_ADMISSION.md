# PROD-Q-R1-R7-D1-R6 — Transaction-Coordinated Serving Admission

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (implementation evidence — not qualification closure) |
| **Parent** | **PROD-Q-R1-R7-D1** / **D1b** — remain **BLOCKED** until independent audit |
| **ADR** | **ADR-01-R6** — **DEC-09** / **DEC-09-EXT** |
| **Scope** | `ActivationDomainCoordinator`, grant registry removal, composition topology validation |

## Evidence summary

- **Coordinator:** `intergrax/agent_distribution/activation_domain_coordinator.py` — in-memory + SQLite `BEGIN IMMEDIATE` snapshot; physical reserve/release with compensation on failure after reserve.
- **Grant authority removed:** no `_PENDING_GRANT_ISSUANCES`, nonce consumption, or `_ACTIVATION_CRITICAL_DEPTH` authorization proof.
- **Adapter:** `authorize_serving_execution_admission` requires coordinator unit-open context + sealed eligibility evidence.
- **Applications:** store-backed per-execution admission and handoff authorize through coordinator; accessor direct admit removed; `validate_colocated_serving_admission_topology`.
- **Tests:** `test_prod_q_r1_r7_d1_r6_activation_domain_coordinator.py` + updated D1/D1b/R2/R3/R4/adapter regressions (**53** targeted tests pass @ implementation commit).

## Explicit non-claims

- Does **not** close **PROD-Q-R1**, **PROD-Q-R1-R7**, **D1**, **D1b**, or any **FRZ-PRD** row.
- Cross-process SQLite writer resistance remains **outside** OPTION A operational trust boundary.
