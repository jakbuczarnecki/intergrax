# NPSC-5F/R4 — Reconstruction Quality (Implementation)

> **Status:** **IMPLEMENTATION COMPLETE** — formal R4 Final freeze is a separate qualification step (`NPSC_5F_R4_FINAL_*`).

## Purpose

Deliver a **read-only** historical view of execution so operators can answer *what happened, in what order, with what decisions* — without replay, side effects, or mutating durable evidence.

```text
history reconstruction != execution replay
```

## Provenance

| Layer | Status | Role |
| ----- | ------ | ---- |
| R1 Durable Evidence | FROZEN | `RuntimeEventPersistence` = source of truth |
| R2 Journal Completeness & Ordering | FROZEN | `ExecutionEventPosition`, positioned journal |
| R3 Governed Evidence Export | FROZEN | Export is not reconstruction authority |
| R4 As-of / bitemporal composition | IMPLEMENTATION COMPLETE | `HistoricalReconstructionService` |

## Reconstruction model

```text
Stored Evidence (RuntimeEventPersistence + CausalEvidencePersistence)
        |
        v
Journal reader / as-of projection (R2)
        |
        v
Reconstruction projection (derived, immutable)
        |
        v
Historical execution view (read-only)
```

No new store, no event bus, no executor path in reconstruction modules.

## Canonical surfaces

| Concern | Path |
| ------- | ---- |
| Execution-level reconstruction (attempts, events, completeness) | `intergrax/runtime/diagnostics/execution_reconstruction.py` |
| E + K + bitemporal historical composition | `intergrax/runtime/observability/historical_reconstruction.py` |
| Historical coordinates & requests | `intergrax/contracts/historical_reconstruction.py` |

### View contracts (existing names)

| User-facing concept | Type |
| ------------------- | ---- |
| Reconstructed execution view | `ExecutionReconstruction` |
| Reconstructed attempt view | `ReconstructedAttempt` |
| Reconstructed event view | `PositionedRuntimeEvent` (within attempts / run scope) |
| Composed historical view | `ExecutionHistoricalReconstruction` |

All views are **frozen dataclasses** — not persisted, not authoritative for live execution.

## Quality guarantees

| Guarantee | Mechanism |
| --------- | --------- |
| Ordering | `ExecutionEventPosition` via `load_positioned_run_journal_through` / positioned lists — not timestamp-only sort |
| Completeness | `RuntimeHistoryCompleteness` (`complete` / `truncated`); truncated prefix → explicit incomplete state |
| Missing / corrupt evidence | `ExecutionReconstructionIntegrityError`, `HistoricalEvidenceIntegrityError` — fail closed, no guess fill |
| Determinism | Same evidence set + boundaries → identical `ExecutionReconstruction` / historical view |
| Tenant isolation | Scoped reads on persistence; cross-tenant requests rejected at contract boundary |
| Read-only | Reconstruction only calls persistence **read** APIs; no append/mutate on evidence stores |

## As-of reconstruction

Optional `execution_as_of: AsOfBoundary` on `ExecutionReconstructor.reconstruct_execution` and `HistoricalReconstructionService.reconstruct` limits the run journal prefix without changing current execution state.

## Qualification gate

`tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_quality.py`

Regression: R1/R2/R3 Final architecture suites (unchanged contracts).

## Related documentation

- R4 as-of / bitemporal implementation: `NPSC_5F_R4_RECONSTRUCTION_ASOF_BITEMPORAL.md`
- Architecture hub: `NPSC_5F_EXECUTION_EVIDENCE_REPLAY_OBSERVABILITY_ARCHITECTURE.md`

## Next step

**NPSC-5F/R4 Final Qualification** — freeze reconstruction quality + historical composition under protected drift policy.
