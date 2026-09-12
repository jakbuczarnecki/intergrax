# NPSC-5F Final — Evidence Plane Qualification and Freeze

**Status:** `FROZEN / PASS`

**Task:** NPSC-5F Final — Evidence Plane Complete Qualification and Freeze

---

## Executive summary

The **Evidence Plane** is the platform read/write boundary for durable execution evidence: canonical `RuntimeEvent` identity, tenant-scoped persistence, ordered journal projection, governed export, and read-only historical reconstruction. It does **not** own execution lifecycle, scheduling, retry/resume, governance decisions, or minting of run/attempt authority.

Enterprise safety follows from: immutable append-only evidence, fail-closed mandatory persistence, export allowlisting, deterministic reconstruction at explicit as-of / bitemporal coordinates, and static proof that evidence roots do not invoke execution-control APIs.

**Production code changed in this Final task:** NO (qualification, drift classifier, regression matrix, gates, documentation only).

**Agent/CI constraint:** Run the mandatory regression matrix via **one** `uv run pytest` invocation (`testing_support/npsc5f_final_regression_matrix.py`). Do **not** fan out dozens of parallel `uv`/`python` subprocesses (process explosion / host memory risk).

---

## Architecture ownership

| Component | Responsibility | Forbidden responsibility |
| --------- | -------------- | ------------------------ |
| `ExecutionRuntime` | Execution lifecycle | Durable evidence ownership |
| `RuntimeEvent` | Event identity & schema | Persistence policy / export redaction |
| `RuntimeEventPersistence` | Durable evidence commit | Journal ordering semantics |
| `UnifiedRunJournal` | Journal projection & completeness | Export envelope authority |
| `ObservabilityExportEnvelope` | Governed export boundary | Historical coordinate binding |
| `HistoricalReconstructionService` | Read-only reconstruction | Mutation of stores / lineage |
| `ExecutionLineagePersistence` | Lineage records | Event store duplication |
| `RuntimeCheckpoint` / `TaskCheckpointPersistence` | Checkpoint durability | Evidence export |
| Canonical Governance Plane | Policy decisions | Evidence persistence |
| Canonical Authority Plane | Authority minting | Evidence projection |

Static negative proof: `testing_support/npsc5f_final_evidence_plane_ownership.py` — Evidence Plane AST roots must not call `retry`, `resume`, `schedule`, bare `execute()`, `mint_attempt`, `mint_run`, `approve`, `deny`, or `mutate_execution`.

Enterprise certification gates (execution / reconstruction / persistence isolation, single durable append path): `tests/unit/runtime/architecture/test_npsc5f_enterprise_evidence_certification.py`. Architecture freeze narrative: `docs/project/architecture/OBSERVABILITY.md` § Evidence Plane freeze (NPSC-5F enterprise certification).

---

## Frozen contracts

| Contract | Status |
| -------- | ------ |
| Durable evidence (R1) | FROZEN |
| Tenant isolation (R1) | FROZEN |
| Event identity (R1) | FROZEN |
| Journal completeness (R2) | FROZEN |
| Export safety (R3) | FROZEN |
| Reconstruction semantics (R4) | FROZEN |
| As-of semantics (R4) | FROZEN |
| Bitemporal semantics (R4) | FROZEN |

Predecessor freezes: NPSC-5E Final, NPSC-5F/R1–R4 Final (see provenance SHAs).

---

## Drift qualification (tri-classifier)

Module: `testing_support/npsc5f_final_evidence_plane_drift.py`

| Class | Meaning |
| ----- | ------- |
| `QUALIFIED_COMPATIBLE` | Qualification/docs/testing_support changes for NPSC-5F |
| `UNRELATED` | Outside Evidence Plane protected surfaces (execution, diagnostics, applications, …) |
| `BREAKING` | Protected production path changed without qualification |

**Protected production surfaces (scoped — not whole runtime):**

- `intergrax/runtime/events/**`
- `intergrax/runtime/observability/**`
- `intergrax/contracts/historical_reconstruction.py`
- `intergrax/contracts/runtime_event.py` (reserved pin; taxonomy lives under `runtime/events/`)

Sentinel: `tests/unit/testing_support/test_npsc5f_final_protected_drift.py`

---

## End-to-end evidence lifecycle gate

`tests/unit/runtime/architecture/test_npsc5f_final_evidence_plane_qualification.py`

Flow: bus record → durable persistence → journal load → governed export → `HistoricalReconstructionService.reconstruct()` → audit projection.

Proves: identity continuity (`tenant_id`, `run_id`, `execution_id`, `attempt_id`, `task_id`, `event_id`), export/persistence alignment, and zero store/causal/knowledge-binding mutation after reconstruction.

---

## Regression evidence

Executed in **one** pytest subprocess: `test_npsc5f_final_mandatory_regression_matrix_passes` (matrix definition in `testing_support/npsc5f_final_regression_matrix.py`).

Covers: Recovery, NPSC-5E/5D Final, HITL R3, child execution, checkpoint, retry, cancellation, evidence (P0 + R1–R4), DG_001, TRACE-ASOF, TRACE-BITEMP, execution reconstruction.

---

## Provenance SHA record

| Label | SHA |
| ----- | --- |
| NPSC-5E FINAL | `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` |
| NPSC-5F/R1 FINAL | `455c09f342f995ac0a6fcb03ffef2f4d3e36a447` |
| NPSC-5F/R2 FINAL | `76c92847f67da22d97943b55896a88c814d7e39d` |
| NPSC-5F/R3 FINAL | `0346face3ef68d8f21504822a26f8f45f2384cf9` |
| NPSC-5F/R4 FINAL (drift pin) | `37fb051c7f164d705f628760436b8ea10ee0289f` |
| Evidence Plane production baseline (R4 implementation) | `3bec620ab56417a469487347f68045bf3dec6bd5` |
| **NPSC-5F FINAL sign-off** | Recorded on merge commit containing this document |

---

## Enterprise guarantees

| Guarantee | Evidence |
| --------- | -------- |
| Deterministic reconstruction | R4 gate + Final E2E |
| Auditability | Journal + export envelope + reconstruction audit refs |
| Replay safety | Read-only reconstruction; no new events/checkpoints/lineage |
| Tenant isolation | R1 Final + export tenant gates |
| Immutable evidence | Idempotent `event_id` append contract |
| No execution side effects | Static negative scan + P0/R4 behavioral proofs |

---

## Definition of Done

```text
STATUS: PASS
NPSC-5F: FROZEN
EVIDENCE PLANE: CERTIFIED
PRODUCTION CODE CHANGED: NO
SECOND EVIDENCE PATH: NO
SECOND STORE: NO
SECOND EXPORT: NO
SECOND RECONSTRUCTION: NO
```

Regression: `test_npsc5f_final_mandatory_regression_matrix_passes` + `test_npsc5f_final_qualification_gate`.
