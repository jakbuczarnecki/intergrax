# DG-001D — Real Supervisor Pre-Engine Failure Qualification (R4)

**Verdict:** PASS

**Date:** 2026-09-07

**Branch:** `development`

**START_HEAD:** `cc1e88385ae2b2f338ef965f74da01ca6c209c0a`

**Task:** `DG-001D-REAL-SUPERVISOR-PRE-ENGINE-FAILURE-QUALIFICATION-R4`

---

## 1. Verdict

```text
DG-001D REAL SUPERVISOR PRE-ENGINE FAILURE QUALIFICATION R4 = PASS
```

R4 proves real process boundaries, durable Mongo persistence, configured Elasticsearch observability export, and cross-process `DiagnosticReadService` read for the R2/R3 supervisor pre-engine `APPLICATION_FAILED` path.

**DG-001D status after R4:**

```text
PRODUCER + HOST-DIAG-3 CONFORMANCE + REAL INTEGRATION QUALIFIED
```

**DG-001 overall:** `PARTIALLY ADDRESSED` (unchanged — DG-001A/C remain open).

---

## 2. R3 vs R4

| Aspect | R3 | R4 |
|--------|----|----|
| Supervisor | Real `HostedApplicationSupervisor.run()` in unit test | Real supervisor in **separate OS process** |
| Engine factory fault | Proof-owned failing factory in test | Proof-owned failing factory in child process |
| Observability | `InMemoryObservabilityExporter` | **Elasticsearch** (`LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND=elasticsearch`) |
| Problem persistence | `InMemoryProblemPersistence` | **MongoDB DocumentStore** (replica-set) |
| Operator read | Same in-memory process | **Separate reader subprocess** + `DiagnosticReadService` |
| Recurrence | Two supervisor runs in same test process | Two independent supervisor processes → `1 Problem / 2 occurrences` |

R3 canonical commit: `70c342d4ccd0d65299bf433fbf7b74818e9f1b45`

---

## 3. Topology

```text
PROCESS A (supervisor child)
  → python scripts/proof/dg001d_r4_supervisor_child.py
  → canonical LKW production authority + Mongo DocumentStore
  → build_local_workspace_worker_bootstrap_diagnostics (HOST-DIAG-3 + Elasticsearch export)
  → HostedApplicationSupervisor + ControlledFailingHostedApplicationEngineFactory
  → APPLICATION_FAILED → observability export → DiagnosticOrchestrator → Mongo Problem
  → exit code 1 (SUPERVISOR_ERROR preserved)

PROCESS B (reader child)
  → python scripts/proof/dg001d_r4_supervisor_failure_qualification.py --mode reader
  → rebuild settings/profile/DocumentStore/DiagnosticReadService
  → list_problems + get_problem from durable Mongo
```

Recurrence (same logical R4 attempt):

```text
PROCESS A1 (instance-001) → 1 Problem / 1 occurrence
PROCESS A2 (instance-002) → same Problem / 2 occurrences (reader confirms)
```

---

## 4. Environment

| Component | Backend |
|-----------|---------|
| DocumentStore | MongoDB replica-set (`mongodb://127.0.0.1:27017/?replicaSet=rs0&directConnection=true`) |
| Database | `intergrax_dg001d_r4` (per-attempt isolated collection) |
| Observability export | Elasticsearch (`http://127.0.0.1:9200`, index `intergrax-lkw-observability`) |
| Redis | Ephemeral local `redis://127.0.0.1:6379/0` (composition prerequisite) |

Harness flags: `--ensure-redis --ensure-mongo-replica-set`

**Prerequisite:** `uv sync --extra dev` (pymongo for Mongo DocumentStore adapter).

Canonical qualification attempt: `dg001d-r4-20260907145456` (all E0–E9 PASS, recurrence `1 Problem / 2 occurrences`).

Evidence artifacts: `.tmp/session/dg001d-r4/<attempt-id>/`

---

## 5. Controlled fault

| Field | Value |
|-------|-------|
| Boundary | Pre-engine `engine_construction` |
| Mechanism | `ControlledFailingHostedApplicationEngineFactory` at public `HostedApplicationEngineFactory` seam |
| Exception | `RuntimeError("DG001D-R4-SECRET-SENTINEL")` |
| Bounded persisted facts | `phase=engine_construction`, `reason_code=engine_factory_failed`, `exception_type=HostedApplicationSupervisorError`, `process_role=hosted_application_supervisor` |
| Raw secret in canonical state | **NO** |
| Production fault flag | **NONE** |

---

## 6. E0–E9 gates

| Gate | Status | Evidence |
|------|--------|----------|
| E0 environment | **PASS** | Mongo replica-set, Elasticsearch, Redis reachable |
| E1 real supervisor process | **PASS** | Subprocess `dg001d_r4_supervisor_child.py` |
| E2 supervisor identity | **PASS** | `application_id`, `instance_id`, tenant bound |
| E3 controlled pre-engine failure | **PASS** | exit 1 / `SUPERVISOR_ERROR` |
| E4 canonical failure event | **PASS** | `hosting.application.failed`, `lifecycle_state=failed` |
| E5 observability export | **PASS** | Elasticsearch `_search` readback |
| E6 diagnostic projection | **PASS** | Durable Problem via HOST-DIAG-3 |
| E7 durable persistence | **PASS** | Mongo replica-set DocumentStore |
| E8 operator read + recurrence | **PASS** | Separate reader subprocess; `1 Problem / 2 occurrences` |
| E9 failure semantics / safety | **PASS** | `APPLICATION_INSTANCE`; no execution identity; sentinel absent |

---

## 7. Contracts confirmed

- `APPLICATION_FAILED` via real supervisor path
- `APPLICATION_INSTANCE` subject
- No Task / Run / Execution identity
- Observability before diagnostics (production HOST-DIAG-3 publisher)
- Raw sentinel not in persisted canonical state
- Supervisor failure truth preserved (non-zero exit)
- No duplicate pre-engine Problem per attempt
- Cross-process durable Mongo read
- Recurrence: `1 Problem / 2 occurrences`

---

## 8. Production diff gate

```text
intergrax/     = NONE
applications/  = NONE
```

Changes limited to `scripts/proof/`, tests, and qualification documentation.

---

## 9. Entrypoints

```powershell
python scripts/proof/dg001d_r4_supervisor_failure_qualification.py `
  --ensure-redis --ensure-mongo-replica-set
```

Child process: `python scripts/proof/dg001d_r4_supervisor_child.py`

---

## 10. Focused regression

| Suite | Scope |
|-------|-------|
| DG-001D R4 unit helpers | `tests/unit/scripts/proof/test_dg001d_r4_supervisor_failure_qualification.py` |
| DG-001D R3 conformance | `tests/unit/hosting/supervisor/test_supervisor_host_diag_3_conformance.py` |
| Supervisor + diagnostics | hosting supervisor + HOST-DIAG-3 integration |

---

## 11. Non-claims

R4 does **not** close DG-001A (default runner wiring) or DG-001C (public launcher bootstrap).
