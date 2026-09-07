# DG-001B — Real Controlled Bootstrap Failure Qualification (R5)

**Verdict:** QUALIFIED

**Date:** 2026-09-07

**Branch:** `development`

**START_HEAD (session):** `2104e4b463bd3d2c9b108037825d72bfde3339e8`

**Pre-commit HEAD:** `67f853a59b446c8aefaeee86217f1c81a6acf687`

**Qualification attempt:** `dg001b-r5-a-20260907134000`

---

## 1. Verdict

```text
DG-001B REAL CONTROLLED BOOTSTRAP FAILURE QUALIFICATION R5 = QUALIFIED ✅
```

All gates **E0–E9 PASS**. Identity fidelity **100%**.

---

## 2. Topology

```text
PROCESS A (parent harness)
  → subprocess: python -m local_workspace_application.host.background_worker_main
  → real production composition (authority, DocumentStore, HOST-DIAG-3, Elasticsearch export)
  → controlled B6 TypeError at create_kafka_worker seam
  → APPLICATION_FAILED exported + DiagnosticOrchestrator projection
  → durable Problem/occurrence in Mongo replica-set DocumentStore
  → exit code 1 (original failure preserved)

PROCESS B (parent harness reader subprocess)
  → rebuild settings/profile/DocumentStore/DiagnosticReadService
  → DiagnosticReadService.list_problems + get_problem
  → operator-visible APPLICATION_INSTANCE occurrence
```

LKW is the real carrier process only; diagnostic core remains product-agnostic.

---

## 3. Environment

| Component | Backend |
|-----------|---------|
| DocumentStore | MongoDB replica-set (`mongodb://127.0.0.1:27017/?replicaSet=rs0&directConnection=true`) |
| Database | `intergrax_dg001b_r5` (per-attempt isolated collection) |
| Observability export | Elasticsearch (`http://127.0.0.1:9200`, index `intergrax-lkw-observability`) |
| Redis | Ephemeral local `redis://127.0.0.1:6379/0` (qualification harness) |
| Kafka bootstrap | `127.0.0.1:9094` (composition only; B6 fails before worker.start) |

Harness flags: `--ensure-redis --ensure-mongo-replica-set`

Evidence artifacts: `.tmp/session/dg001b-r5/dg001b-r5-a-20260907134000/`

---

## 4. Controlled fault

| Field | Value |
|-------|-------|
| Boundary | **B6 / `worker_construction`** |
| Mechanism | `LOCAL_WORKSPACE_WORKER_CONSTRUCTION_FAULT=typed_bootstrap_exception` → `maybe_raise_worker_construction_fault()` immediately before `create_kafka_worker(...)` |
| Exception | `TypeError("create_kafka_worker composition failure DG001B-R5-SECRET-SENTINEL")` |
| Bounded persisted facts | `phase=worker_construction`, `reason_code=bootstrap_unhandled_exception`, `exception_type=TypeError`, `process_role=background_worker` |
| Raw secret in canonical state | **NO** |

---

## 5. E0–E9

| Gate | Status | Evidence |
|------|--------|----------|
| E0 environment | **PASS** | Mongo replica-set, Elasticsearch, Redis reachable |
| E1 real worker process | **PASS** | Subprocess `background_worker_main` |
| E2 bootstrap identity | **PASS** | `application_id`, `instance_id`, tenant bound |
| E3 controlled B6 failure | **PASS** | Deterministic TypeError at worker construction seam |
| E4 canonical failure event | **PASS** | `hosting.application.failed`, `lifecycle_state=failed` |
| E5 observability export | **PASS** | Elasticsearch document readback (HTTP 201 + `_search`) |
| E6 diagnostic projection | **PASS** | Durable Problem via HOST-DIAG-3 |
| E7 durable persistence | **PASS** | Mongo replica-set DocumentStore |
| E8 independent operator read | **PASS** | Separate reader subprocess + `DiagnosticReadService` |
| E9 failure semantics / safety | **PASS** | exit 1; no execution identity; sentinel absent |

---

## 6. Identities (attempt `dg001b-r5-a-20260907134000`)

| Field | Value |
|-------|-------|
| `application_id` | `local_workspace` |
| `instance_id` | `6ad84d74-ea83-421a-bb25-3490356dd3eb` |
| diagnostic tenant | `local_workspace.product` |
| `event_id` | `evt-199d812a-0a63-46ba-b7b2-4fcf7bc6c510` |
| `ProblemId` | `problem_0fd694c1e1244c97b1eaa5b14bb37dd6` |
| occurrence id | `application_instance:local_workspace:6ad84d74-ea83-421a-bb25-3490356dd3eb` |
| TaskId / RunId / AttemptId / ExecutionId | **NONE** |
| subject kind | `APPLICATION_INSTANCE` |
| identity fidelity | **100%** |
| child exit code | **1** |

---

## 7. Non-claims

R5 does **not** prove supervisor pre-engine failures, public launcher/bootstrap failures, config failures before B5, or universal hosting coverage.

R5 **does** prove Central Diagnostics can diagnose a real pre-execution hosted worker bootstrap failure after diagnostic prerequisites B3–B5 exist, without execution identity.

---

## 8. Repeatability

Canonical attempt **R5-A** = PASS above.

**R5-B** repeat run: **NOT RUN** (single immutable canonical attempt satisfied all gates; next repeat requires new attempt id).

---

## 9. Regression

| Suite | Count |
|-------|------:|
| R2/R3/R4 + harness unit tests | 38 passed |
| `--ignore` | NO |
| new skips | NONE |

---

## 10. Final statement

```text
DG-001B = REAL PRE-EXECUTION WORKER BOOTSTRAP DIAGNOSTICS QUALIFIED ✅
DG-001 = PARTIALLY ADDRESSED
NEXT = DG-001B-FINAL-CLOSURE-REVIEW-R6
```
