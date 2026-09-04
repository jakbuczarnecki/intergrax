# DIAG-FUNCTIONAL-READ-R1-R1 Qualification

**Verdict:** PASS

**Date:** 2026-09-04

**Branch:** `development`

**S1 baseline qualified SHA:** `98caff3af51b2951b8f0704ac7f96fea526cbfd5`

**R1-R1 final qualified SHA:** see git `development` after R1-R1 merge (record at commit time)

## Independent audit defect (pre-R1-R1)

`FunctionalEvidenceIndexRebuilder.ensure_v2_projection()` used:

```text
if ANY execidx:v2 row exists → assume projection complete → skip rebuild
```

Crash mid-rebuild (e.g. 137/5000 v2 rows written) left partial v2 navigable on restart → silent evidence loss.

## Architecture options

| Option | Summary | Verdict |
| ------ | ------- | ------- |
| A — projection completeness manifest | `execidxmeta:{task}:{run}` with `building` → `complete`; O(1) fast path when `complete` | **Selected** |
| B — full v1 reconcile on every ensure | Correct but O(E) every query | Rejected — destroys R1 hot path |

**Selected:** Option A (`intergrax.functional_evidence.projection_state.v1`) plus fixed-point v1→v2 reconciliation before conditional `complete` transition; orphan v2 verification; v2-before-v1 append ordering.

## Projection completeness model

| Layer | Role |
| ----- | ---- |
| `record:{evidence_id}` | **Canonical truth** |
| `exec:{task}:{run}:{evidence_id}` | Derived legacy v1 projection (migration/repair source) |
| `execidx:{task}:{run}:{micros}:{evidence_id}` | Derived v2 query projection |
| `execidxmeta:{task}:{run}` | Derived migration/control metadata — **not truth** |

**Invariant:** partial v2 ≠ complete. Query uses v2 only when manifest `state=complete`.

## Failure-mode matrix

| Failure point | Persisted state | Pre-R1-R1 | Post-R1-R1 |
| ------------- | --------------- | --------- | ------------ |
| canonical before v1/v2 | canonical only | silent omission on query | silent omission until append retry (canonical scan not used on query path) |
| v2 before v1 (append crash) | canonical + v2 | N/A (v1 first) | query OK via v2; v1 repaired on append retry |
| v1 before v2 (legacy partial rebuild) | partial v2, no complete manifest | **silent subset** | reconcile on ensure |
| rebuild before COMPLETE marker | BUILDING + partial v2 | treated complete if any v2 | reconcile resumes |
| after COMPLETE | manifest complete + indexes | fast path | O(1) manifest check |
| concurrent append + rebuild | v1+v2 from append + BUILDING | race window | fixed-point reconcile + append writes both indexes |
| orphan v2 | v2 without v1/canonical | undetected if no v1 | fail closed on ensure |
| corrupt v2 metadata | mismatch vs canonical | fail on read | fail closed on reconcile/read |

## Process restart proof (real Mongo)

Profile (frozen):

- E = 1000 legacy v1-only evidence
- interrupt after 137 v2 writes
- page_size = 25
- PROCESS A: seed + interrupted rebuild
- PROCESS B: subprocess `recovery_reader_probe.py`, fresh Mongo adapter

| Metric | Value |
| ------ | ----- |
| rows written before interrupt | 137 |
| recovered_count (PROCESS B) | **1000** |
| passed | **true** |

Artifact: `.tmp/proof/diag-functional-read-r1r1/mongo-recovery-qualification.json`

## R1 hot-path regression (Mongo)

Command: `uv run python -m tests.system.functional_diagnostics_read_r1.mongo_qualification`

| Metric | Value |
| ------ | ----- |
| E | 5000 |
| P | 25 |
| first_page canonical gets | 26 |
| union_count | 5000 |
| passed | true |

## Unit proofs

`tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r1_projection_recovery.py` — 12 cases:

1. v1-only → complete migration
2. partial v2 → repair
3. partial v2 + interrupted rebuild → repair after fresh adapter
4. complete manifest → no full rebuild
5. concurrent rebuild
6. append after rebuild
7. canonical+v1 missing v2 → repair
8. corrupt v2 metadata → fail closed
9. orphan v2 → fail closed
10. missing canonical → fail closed
11. filtered pagination after recovery
12. cursor union after recovery

## Regressions

| Gate | Result |
| ---- | ------ |
| R1 bounded unit proofs | PASS |
| R1-R1 projection recovery unit | PASS |
| D1 durable conformance unit | PASS |
| R1 Mongo hot-path | PASS |
| R1-R1 Mongo process recovery | PASS |

## Files changed

- `intergrax/runtime/diagnostics/functional_evidence_projection_state.py` (new)
- `intergrax/runtime/diagnostics/functional_evidence_index_rebuilder.py`
- `intergrax/runtime/diagnostics/document_store_functional_evidence_persistence.py`
- `tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r1_projection_recovery.py` (new)
- `tests/system/functional_diagnostics_read_r1_r1/` (new)
- `docs/project/architecture/DIAGNOSTICS.md`
- `docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_QUALIFICATION.md`

## Final architecture statement

```text
PARTIAL FUNCTIONAL EVIDENCE PROJECTION = NEVER TREATED AS COMPLETE
FUNCTIONAL EVIDENCE PROJECTION RECOVERY = CRASH-SAFE AND IDEMPOTENT
CANONICAL FUNCTIONAL EVIDENCE = SOLE SOURCE OF TRUTH
HEALTHY FUNCTIONAL EVIDENCE READ PATH = PAGE-BOUNDED / INCREMENTAL
```
