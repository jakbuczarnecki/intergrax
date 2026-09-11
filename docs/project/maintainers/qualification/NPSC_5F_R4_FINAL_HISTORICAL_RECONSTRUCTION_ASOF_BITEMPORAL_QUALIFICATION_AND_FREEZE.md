# NPSC-5F/R4 Final — Historical Reconstruction As-Of Bitemporal Qualification and Freeze

**Status:** `FROZEN / PASS`

**Task:** NPSC-5F/R4 Final — Historical Reconstruction As-Of Bitemporal Qualification and Freeze

---

## Purpose

Formal enterprise freeze for read-only historical reconstruction: explicit **E** (`AsOfBoundary` / `ExecutionEventPosition`), **K** (`KnowledgeRevisionWatermark`), and bitemporal query (`BitemporalKnowledgeBasis`) coordinates compose one canonical `HistoricalReconstructionService.reconstruct()` path without execution, recovery, governance, or export-authority ownership.

Invariant:

> **Historical reconstruction answers what execution and knowledge were visible at declared boundaries; it never mutates durable evidence, never mints run/attempt identity, and never orders execution by timestamp alone.**

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
| **R4 implementation baseline** | `3bec620ab56417a469487347f68045bf3dec6bd5` |
| **R4 Final qualification sign-off** | Recorded on merge commit containing this document |

**Production code changed in R4 Final task:** NO (qualification, drift helper, regression matrix runner, final gate, docs only).

---

## Canonical components (frozen contracts)

| Area | Component |
| ---- | --------- |
| Contract | `intergrax/contracts/historical_reconstruction.py` |
| Runtime service | `HistoricalReconstructionService` |
| Reconstruction model | `ExecutionHistoricalReconstruction` |
| Execution projection | `ExecutionReconstruction` (DIAG consumption) |
| Knowledge projection | `HistoricalKnowledgeProjection` |
| Valid time | `AsOfBoundary` |
| Knowledge time | `KnowledgeRevisionWatermark` |
| Bitemporal basis | `BitemporalKnowledgeBasis` |
| Ordering | `ExecutionEventPosition` |

**R4 FREEZE = current canonical contracts.** Deferred (not added in R4 Final): `EvidenceProjection`, `projection_trace`, `EvidenceCompleteness` as separate public contracts — explainability remains `ExecutionReconstruction`, `RuntimeHistoryCompleteness`, and reconstruction `limitations`.

---

## R4 ownership (allowed)

| Responsibility | Owner |
| -------------- | ----- |
| Historical coordinate binding | `HistoricalReconstructionBasis` / `ExecutionHistoricalReconstructionRequest` |
| Read-only composition | `HistoricalReconstructionService` |
| Execution prefix at E | `load_positioned_run_journal_through` (R2) |
| Knowledge at K | `reconstruct_knowledge_at_watermark` (TRACE-BITEMP) |
| Bitemporal admissibility | `revision_admissible_at_bitemporal_query` |
| Fail-closed integrity | `HistoricalEvidenceIntegrityError`, `KnowledgeBoundaryNotFinalizedError` |

---

## Forbidden ownership (R4 MUST NOT)

| Forbidden action | R4 stance |
| ---------------- | --------- |
| Mint `RunId` / `AttemptId` | Static + behavioral proof — absent |
| Mutate lineage | Read-only ports only |
| Retry / resume / scheduler | No control-plane symbols on R4 surfaces |
| Governance decision | Out of scope |
| Export envelope as authority | No `ObservabilityExportEnvelope` / `JournalExportSnapshot` on reconstruction service |

---

## Acceptance criteria (evidence)

| Criterion | Evidence |
| --------- | -------- |
| Events after valid-time E absent from execution projection | `test_r4_execution_boundary_inclusive`, prefix truncation / missing boundary tests |
| Events after system-time K absent from knowledge view | `test_r4_late_knowledge_bitemporal` (and admissibility unit tests) |
| Deterministic reconstruction | `test_r4_determinism_and_clock_independence` |
| `ExecutionEventPosition` ordering (no timestamp-only ordering) | `test_r4_same_timestamp_orders_by_position` |
| Read-only after `reconstruct()` | `test_r4_reconstruct_does_not_mutate_evidence_or_knowledge` |
| Provider parity on execution prefix | `test_r4_provider_parity_execution_prefix` |
| No active execution surface | `test_r4_static_no_active_execution_surface` |

---

## Drift qualification

Sentinel: `test_npsc5f_r4_final_protected_drift` in `tests/unit/testing_support/test_npsc5f_r4_final_protected_drift.py`

Classifier: `testing_support/npsc5f_r4_protected_drift.py` — drift since R4 implementation baseline `3bec620a..origin/development` must remain empty on protected surfaces.

**Protected (post-freeze):**

- `intergrax/contracts/historical_reconstruction.py`
- `intergrax/runtime/observability/historical_reconstruction.py`
- R4 implementation + Final qualification gates and drift helpers (`testing_support/npsc5f_r4_*`)

**Explicitly not file-frozen (parallel work allowed):**

- `intergrax/runtime/events/**` taxonomy / runtime event evolution
- `intergrax/runtime/events/asof_projection.py`
- `intergrax/runtime/diagnostics/execution_reconstruction.py`
- `intergrax/contracts/bitemporal_knowledge.py` (shared TRACE-BITEMP contract)

Regression matrix runs in **one** `uv run pytest` invocation (`testing_support/npsc5f_r4_regression_matrix.py`) with orchestrator tests deselected — avoids nested subprocess / process explosion.

---

## Regression matrix

Executed via `test_mandatory_regression_matrix_passes` in `test_npsc5f_r4_final_historical_reconstruction_qualification_and_freeze.py` (single subprocess, `-p no:xdist`).

| Suite | Target |
| ----- | ------ |
| R4 implementation gate | `test_npsc5f_r4_reconstruction_asof_bitemporal.py` |
| NPSC-5F/R3 Final | `test_npsc5f_r3_final_governed_evidence_export.py` |
| NPSC-5F/R2 Final | `test_npsc5f_r2_final_journal_completeness_ordering.py` |
| NPSC-5F/R1 Final | `test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py` |
| NPSC-5F P0 | `test_npsc5f_p0_execution_evidence_architecture_reconciliation.py` |
| TRACE-ASOF | `test_execution_position_asof.py`, `test_asof_projection.py` |
| TRACE-BITEMP | `test_bitemporal_revision_ordering.py`, `test_bitemporal_knowledge.py`, `test_knowledge_reconstruction.py` |
| Execution reconstruction | `test_execution_reconstruction.py` |
| DG_001 | `test_execution_lineage_contracts.py`, `tests/unit/runtime/execution/lineage/` |
| NPSC-5E Final | `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` |
| NPSC-5D Final | `test_npsc5d_final_multi_agent_governance_qualification.py` |
| NPSC-5C | `test_npsc5c_coordination_intent_gate.py`, `test_npsc5c_decision_projection_gate.py` |
| NPSC-5B Final | `test_npsc5b_final_production_fanout_fanin_qualification.py` |
| NPSC-5A | `test_npsc5a_multi_agent_coordination_gate.py` |
| R4 Final drift | `test_npsc5f_r4_final_protected_drift.py` |

**Test evidence (local qualification run):** `321 passed`, `36 deselected` (matrix orchestrators), ~28s matrix + R4 gate/drift suites PASS.

---

## Predecessor preservation

- **R1 / R2 / R3 Final gates:** PASS within matrix
- **Recovery plane (5E):** PASS within matrix
- **R3 export ≠ reconstruction source:** `test_r4_no_export_envelope_as_source`

---

## Final verdict

**R4 contract qualification:** PASS on implementation baseline `3bec620a` with **no unqualified R4-protected production drift** since implementation through Final sign-off.

**Historical Reconstruction:** FROZEN R4 — read-only enterprise reconstruction layer for Evidence Plane.

---

## Freeze statement

> Execution and knowledge history at explicit E, K, and bitemporal coordinates are reconstructed only through `HistoricalReconstructionService` and typed contracts in `historical_reconstruction.py`. Reconstruction consumes durable `RuntimeEventPersistence` and finalized knowledge watermarks; it does not create runs, attempts, checkpoints, retries, resumes, scheduler actions, or governance decisions. Run-local order is `ExecutionEventPosition`; timestamp alone is not canonical order. Supported explainability for freeze remains execution reconstruction diagnostics and documented limitations — not a second evidence projection contract family.

**Next:** NPSC-5F Final Certification (whole Evidence Plane)
