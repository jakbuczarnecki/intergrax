# EE-B2 — Chaos Engineering & Fault Injection Enterprise Certification

**Task:** EE-B2  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `471e183d5cd16639a63ced068f2b221be29903c1` |
| **START_ORIGIN** | `471e183d5cd16639a63ced068f2b221be29903c1` |

## Owner decision

**NO NEW CHAOS RUNTIME** — fault injection is test/qualification-only via existing ports (`ExecutionWorkPort`, `RuntimeEventPersistence`, capacity admission, checkpoint store, observability exporter). Production ownership unchanged (EE-A1, EE-B1.x, NPSC-5E, NPSC-5F, W5-H1).

## Chaos infrastructure

| Item | Location |
|------|----------|
| Deterministic fault plan | `testing_support/chaos/fault_plan.py` |
| Failing persistence wrapper | `testing_support/chaos/failing_persistence.py` |
| Port fault adapters | `testing_support/chaos/execution_ports.py` |
| Phase barriers | `testing_support/chaos/barriers.py` |
| Architecture model | `docs/project/maintainers/architecture/EXECUTION_ENGINE_CHAOS_ENGINEERING_AND_FAULT_INJECTION_MODEL.md` |

## Scenario certification

| Scenario | Module | Terminal state |
|----------|--------|----------------|
| Worker failure | `test_ee_b2_worker_fault_injection.py` | FAILED / SUCCEEDED siblings |
| Dependency failure | `test_ee_b2_dependency_fault_injection.py` | DEPENDENCY_FAILURE + typed ConnectionError |
| Dependency timeout | same | TimeoutError + sibling SUCCESS |
| Capacity saturation | `test_ee_b2_capacity_saturation_fault.py` | REJECT + release after fault |
| Cancellation | `test_ee_b2_cancellation_fault.py` | CancelledError + permit release |
| Mandatory evidence | `test_ee_b2_evidence_persistence_fault.py` | FAIL_CLOSED |
| Checkpoint | `test_ee_b2_checkpoint_fault.py` | StaleCheckpointWriteError |
| Recovery interruption | `test_ee_b2_recovery_interruption.py` | No corrupt resume |
| Child partial failure | `test_ee_b2_child_partial_failure.py` | Partial FAILED, no replay |
| OTLP export | `test_ee_b2_observability_export_fault.py` | Evidence persisted; export failed |
| Compound | `test_ee_b2_compound_failure.py` | Primary worker/evidence authority |
| Shutdown | `test_ee_b2_shutdown_fault.py` | Phase order + drain cancel |

## Architecture gates

`test_ee_b2_architecture_gate.py` — forbidden chaos runtime symbols, no `testing_support.chaos` in `intergrax/`, no reflection in chaos helpers, no identity mint in chaos helpers.

## Production code changed

**NO** (tests, `testing_support/chaos`, documentation only).

## Final verdict

**PASS** when EE-B2 pytest slice and frozen regression matrix complete with zero new static errors.
