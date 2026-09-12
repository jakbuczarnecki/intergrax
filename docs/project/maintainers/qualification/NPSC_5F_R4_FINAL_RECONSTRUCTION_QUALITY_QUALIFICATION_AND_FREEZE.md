# NPSC-5F/R4 Final — Reconstruction Quality Qualification and Freeze

**Status:** `FROZEN / PASS`

**Task:** NPSC-5F/R4 Final — Reconstruction Quality Qualification and Freeze

---

## Purpose

Formal enterprise freeze for read-only execution reconstruction: durable `RuntimeEventPersistence` facts and optional causal / lineage enrichment compose immutable `ExecutionReconstruction` views without replay, mutation, or execution authority.

Invariant:

> **Reconstruction answers what happened in run-local execution order from persisted evidence; it never appends facts, never schedules work, and never orders history by timestamp alone.**

---

## Provenance

| Label | SHA |
| ----- | --- |
| NPSC-5F/R3 Final | `0346face3ef68d8f21504822a26f8f45f2384cf9` |
| NPSC-5F/R2 Final | `76c92847f67da22d97943b55896a88c814d7e39d` |
| NPSC-5F/R1 Final | `455c09f342f995ac0a6fcb03ffef2f4d3e36a447` |
| NPSC-5F/P0 | `7811371da1069b661987b050a4c9bf42c02bda69` |
| NPSC-5E Final | `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` |

---

## SHA record

| Label | SHA |
| ----- | --- |
| **R4 reconstruction quality implementation** | `84e704eec611e7b24eb82b0be4fe98172c512739` |
| **R4 Final qualification sign-off** | Recorded on merge commit containing this document |

**Production code changed in R4 Final task:** NO (qualification, drift helper, regression matrix, final gate, docs only).

---

## Source of truth

| Layer | Owner |
| ----- | ----- |
| Durable execution facts | `RuntimeEventPersistence` |
| Causal relation facts | `CausalEvidencePersistence` |
| Optional lineage enrichment | `ExecutionLineageReader` (read port) |
| Derived historical view | `ExecutionReconstructor` → `ExecutionReconstruction` |

Reconstruction is **not** persisted and **not** authoritative for live execution.

---

## Frozen invariants

| Invariant | Enforcement |
| --------- | ----------- |
| Immutable output | Frozen dataclasses (`ExecutionReconstruction`, `ReconstructedAttempt`, …) |
| Deterministic | Same evidence + boundaries ⇒ identical reconstruction |
| Read-only | Persistence append/mutate paths not invoked during reconstruct |
| Run-local order | `ExecutionEventPosition` — not timestamp sort |
| Completeness explicit | `RuntimeHistoryCompleteness` (`complete` / `truncated`) |
| Fail closed | `ExecutionReconstructionIntegrityError` on scope corruption |
| Tenant isolation | Cross-tenant reads return empty scoped views |
| No replay | No execution control-plane symbols on reconstruction surfaces |

---

## Protected production surfaces (drift-scoped)

- `intergrax/runtime/diagnostics/execution_reconstruction.py`
- `intergrax/runtime/diagnostics/execution_lineage_reconstruction.py`

Historical as-of / bitemporal composition (`HistoricalReconstructionService`) remains under the separate R4 as-of freeze — not file-frozen by this record.

---

## Qualification evidence

| Gate | Role |
| ---- | ---- |
| `test_npsc5f_r4_reconstruction_quality.py` | R4 reconstruction quality contracts |
| `test_npsc5f_r4_final_reconstruction_quality_qualification_and_freeze.py` | Final freeze gate + regression matrix |
| `test_execution_reconstruction.py` | DIAG-2 behavioral depth |
| R1 / R2 / R3 Final architecture gates | Predecessor evidence plane |
| `test_npsc5f_r4_quality_protected_drift.py` | Production drift sentinel |

Regression runner: `testing_support/npsc5f_r4_quality_regression_matrix.py`.

---

## What this freeze does **not** cover

- `RuntimeEventPersistence` write semantics (R1)
- Journal pagination / snapshot contracts (R2)
- Governed export envelopes (R3)
- `HistoricalReconstructionService` E/K/bitemporal composition (R4 as-of Final)
- Recovery, retry, checkpoint, scheduler, or governance ownership
- New replay engines or execution services

---

## Freeze statement

> Execution history for a tenant-scoped run is reconstructed only through `ExecutionReconstructor` from persisted runtime and causal facts (and optional read-only lineage enrichment). Outputs are immutable, deterministic, and explicitly complete or truncated. Reconstruction does not mutate durable evidence, does not invoke execution control surfaces, and does not establish cross-run or task-global chronology from timestamps alone. `RuntimeEventPersistence` remains the durable source of execution facts.

**Next:** Maintain R4 quality drift sentinel on `development`; coordinate as-of historical changes through the separate R4 as-of protected path set.
