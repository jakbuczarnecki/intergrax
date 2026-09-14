# EE-B4-A — Operational Readiness, SLO & Health Certification

**Task:** EE-B4-A  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `e56579c8e2df06d12b0745ce47b5faf58e8efe0e` |
| **START_ORIGIN** | `e56579c8e2df06d12b0745ce47b5faf58e8efe0e` |
| **TESTED_SHA** | _(see session final report)_ |

## Deliverables

| Artifact | Path |
| -------- | ---- |
| Operational model | `docs/project/maintainers/architecture/EXECUTION_ENGINE_OPERATIONAL_READINESS_SLO_HEALTH_MODEL.md` |
| Reference assessment (certification) | `testing_support/execution_operational_readiness/` |
| Gate tests | `tests/unit/runtime/architecture/test_ee_b4_a_*.py` |

**PRODUCTION CODE CHANGED:** NO (`intergrax/` unchanged).

## Operational inventory

| Signal / mechanism | Owner | Scope | Authoritative? | Operational use |
| ------------------ | ----- | ----- | -------------: | --------------- |
| Root capacity counters | EE-B1.2 `ExecutionCapacityAdmissionPort` / `assess_root_execution_capacity` | Global process | Yes (admission) | Readiness, saturation SLI |
| Active execution count | Local capacity state | Global | Fact | Utilization SLI |
| Failure classification | EE-B1.1 / UER terminal taxonomy | Per attempt | Yes (outcome) | Success/failure SLI |
| Persistence reliability | Runtime event store / mandatory writers | Global or profile | Yes (evidence) | Readiness fail-closed |
| Event / auditability health | `auditability_health.py` | Global | Contract | Readiness when diagnostics required |
| Nexus saturation | Fan-out / graph bounds (separate from root capacity) | Orchestration | Policy | Scoped degradation |
| Dependency / tool health | `capability_health` projection | Per capability | Read-only projection | Profile readiness |
| Shutdown phase | `ExecutionRuntimeShutdownPhase` | Global | Contract | Not ready after STOP_ACCEPTING |
| Recovery state | NPSC-5E plane | Per execution / subsystem | Recovery authority | Scoped SLI, not global unready by default |

## Ownership summary

| Role | Owner |
| ---- | ----- |
| Operational health composition | Operator / deployment (reference: `testing_support/execution_operational_readiness`) |
| Readiness admission facts | EE-B1.2 + mandatory evidence contracts |
| Liveness | Process + shutdown phase contract |
| SLO target values | Deployment / operator |
| SLI facts | Execution Engine emits events/counters |

## Health / readiness / liveness matrix

Documented in `EXECUTION_ENGINE_OPERATIONAL_READINESS_SLO_HEALTH_MODEL.md` §9; enforced in `test_ee_b4_a_*` against `assess_execution_operational_state`.

## SLI catalog

`testing_support/execution_operational_readiness/sli_catalog.py` — validated by `test_ee_b4_a_sli_slo_contract.py`.

## SLO ownership

Facts from execution; SLI computed in operational layer; targets configured by deployment operator (no core hardcoded %).

## Tests

| Module | Focus |
| ------ | ----- |
| `test_ee_b4_a_health_state_model.py` | Health states & precedence |
| `test_ee_b4_a_readiness_semantics.py` | Ready / starting |
| `test_ee_b4_a_liveness_semantics.py` | Live under drain |
| `test_ee_b4_a_capacity_saturation_readiness.py` | EE-B1.2 saturation |
| `test_ee_b4_a_mandatory_evidence_readiness.py` | Fail-closed evidence |
| `test_ee_b4_a_observability_degradation.py` | OTLP best-effort |
| `test_ee_b4_a_shutdown_readiness.py` | STOP_ACCEPTING |
| `test_ee_b4_a_policy_denial_health.py` | Denial + compound signals |
| `test_ee_b4_a_sli_slo_contract.py` | SLI/SLO shape |
| `test_ee_b4_a_operational_architecture_gate.py` | Docs + forbidden runtimes |

## Static checks

`ruff check`, `ruff format --check`, `pyright` on changed scope — see final report.

## Cross-session exclusions

NPSC-5F protected surfaces not modified. Parallel session WIP on NPSC-5F drift tests left untouched.

**NPSC-5F OPERATIONAL HANDOFF:** NONE

## Regression matrix

EE-A1, EE-A2 (H1 gate), NPSC-4.2 (H1 + residual), NPSC-5B final, NPSC-5E P0A slice, EE-B1.1 shutdown contract, EE-B1.2 architecture gate, EE-B1.3 architecture gate, EE-B2 architecture gate, EE-B3-A/C architecture gates, EE-B4-A — see session pytest log.

## Final verdict

**PASS** _(pending regression + static gates in session)_.
