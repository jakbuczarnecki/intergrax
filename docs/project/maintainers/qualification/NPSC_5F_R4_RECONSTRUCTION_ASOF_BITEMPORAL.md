# NPSC-5F/R4 — Reconstruction / As-of / Bitemporal (Implementation)

> **Status:** NPSC-5F/R4 **FROZEN** — see [`NPSC_5F_R4_FINAL_HISTORICAL_RECONSTRUCTION_ASOF_BITEMPORAL_QUALIFICATION_AND_FREEZE.md`](NPSC_5F_R4_FINAL_HISTORICAL_RECONSTRUCTION_ASOF_BITEMPORAL_QUALIFICATION_AND_FREEZE.md)

## Purpose

Provide one canonical, read-only historical reconstruction path that answers what execution state, knowledge state, lineage, and causal evidence were visible at explicit **E** (execution), **K** (knowledge watermark), and **VT/system-time** (bitemporal query) boundaries — without execution, retry, resume, or side effects.

## Provenance

- **R1** durable evidence commit (`RuntimeEventPersistence`)
- **R2** positioned journal prefix (`load_positioned_run_journal_through`)
- **R3** export boundary is **not** a reconstruction source of truth

## R4 ownership

| Surface | Path |
| ------- | ---- |
| Coordinates & request/result contracts | `intergrax/contracts/historical_reconstruction.py` |
| Composition service | `intergrax/runtime/observability/historical_reconstruction.py` |
| Qualification gate | `tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py` |

## Existing components reused

- `reconstruct_run_execution_as_of` / `RunExecutionAsOfProjection` (TRACE-ASOF-2)
- `ExecutionReconstructor.reconstruct_execution` with optional `execution_as_of` (DIAG-2)
- `reconstruct_knowledge_at_watermark` (TRACE-BITEMP-3)
- `RevisionOrderingAuthority` finalized watermark semantics (TRACE-BITEMP-1/2)

## Historical coordinate model

`HistoricalReconstructionBasis` binds **tenant**, **run**, `AsOfBoundary` **E**, `KnowledgeRevisionWatermark` **K**, and `BitemporalKnowledgeBasis` query (valid time + system time). Axes are not collapsed.

## Execution axis E

Run-local `ExecutionEventPosition` via inclusive `AsOfBoundary`; scope **tenant + run**.

## Knowledge axis K

Tenant-scoped `KnowledgeRevisionWatermark` — contiguous finalized prefix only.

## Valid-time / system-time

`revision_admissible_at_bitemporal_query` evaluates valid-time and system-time independently (late knowledge: accepted system time after query → absent).

## Read-only guarantee

`HistoricalReconstructionService` depends only on read ports; no `ExecutionRuntime`, Nexus, tools, or export envelopes as authority.

## Regression matrix

Full matrix: `test_npsc5f_r4_final_historical_reconstruction_qualification_and_freeze.py` + `testing_support/npsc5f_r4_regression_matrix.py` (single `uv run pytest`).

## Scale follow-up

Document-backed full-platform scans beyond existing bounded journal APIs → **SCALE FOLLOW-UP**, owner Session C.

## Final verdict

R4 Final qualification freeze — PASS (read-only historical reconstruction certified).
