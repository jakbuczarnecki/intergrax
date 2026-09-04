# DIAG-FUNCTIONAL-READ-R1-R2 Qualification

**Verdict:** PASS

**Date:** 2026-09-04

**Branch:** `development`

**S1 baseline qualified SHA:** `98caff3af51b2951b8f0704ac7f96fea526cbfd5`

**R1-R2 final qualified SHA:** `46159f15b8f82642e09ae67a04478e31003cdac2`

## Scope split

| Milestone | Defect closed |
| --------- | ------------- |
| R1-R1 | Legacy v1→v2 partial rebuild treated as complete → silent omission |
| **R1-R2** | **Modern append crash after canonical → canonical invisible to v2 query → silent omission** |

## Architecture options

| Option | Summary | Verdict |
| ------ | ------- | ------- |
| A — per-evidence append intent (`appendpending:`) | Durable execution-scoped signal before canonical; repair from canonical; no shared manifest flip per append | **Selected** |
| B — shared manifest COMPLETE→DIRTY | Correct but contends on every concurrent append for same execution | Rejected |
| C — per-execution write journal / watermark | Heavier; serializes repair semantics | Rejected |

**Selected:** Option A — `intergrax.functional_evidence.append_intent.v1` with write order: intent → canonical → v2 → v1 → clear intent.

## Projection / intent model

| Layer | Role |
| ----- | ---- |
| `record:{evidence_id}` | **Canonical truth** |
| `appendpending:{task}:{run}:{evidence_id}` | Derived append-consistency metadata — detect incomplete projection |
| `execidx:{task}:{run}:{micros}:{evidence_id}` | Derived v2 query projection |
| `exec:{task}:{run}:{evidence_id}` | Derived v1 projection (repair / migration source) |
| `execidxmeta:{task}:{run}` | Derived migration manifest (R1-R1) — legacy rebuild completeness only |

**Query fast path:** manifest `complete` **AND** zero unresolved `appendpending` rows (bounded `limit=1` prefix probe) → incremental v2 read.

## Failure-mode matrix

| Crash point | Persisted state | Recovery |
| ----------- | --------------- | -------- |
| after intent | pending only | fail closed on query (consistency pending); intent never reclaimed by reader; writer retry completes append |
| after canonical | pending + canonical | repair v2 + v1 from canonical; clear intent |
| after v2 | pending + canonical + v2 | repair v1; clear intent |
| after v1 | pending + canonical + v2 + v1 | verify; clear intent |
| after intent clear | healthy | O(1) manifest + pending probe; v2 fast path |

## Process restart proof (real Mongo)

Profile (frozen):

- E = 1000 healthy appended evidence
- 1 new append with fault after canonical
- page_size = 25
- PROCESS A: writer with fault injector
- PROCESS B: subprocess `recovery_reader_probe.py`, fresh Mongo adapter

| Metric | Value |
| ------ | ----- |
| expected_count | **1001** |
| recovered_count | **1001** |
| passed | **true** |

Crash-matrix subprocess scenarios (5 base + 1 crashed append each):

| boundary | expected | recovered | passed |
| -------- | -------- | --------- | ------ |
| after_intent | 5 | 5 | true |
| after_canonical | 6 | 6 | true |
| after_v2 | 6 | 6 | true |
| after_v1 | 6 | 6 | true |

Artifact: `.tmp/proof/diag-functional-read-r1r2/mongo-recovery-qualification.json`

## Unit proofs

`tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r2_append_crash_safety.py` — 15 cases:

1. crash after intent
2. crash after canonical
3. crash after v2
4. crash after v1
5. healthy append
6. retry same evidence
7. conflict same ID different payload
8. concurrent retry same evidence
9. concurrent different evidence
10. query while pending
11. corrupt intent
12. corrupt v2
13. manifest complete + pending intent
14. healthy fast-path operation count
15. cursor union after repair

## Regressions

| Gate | Result |
| ---- | ------ |
| R1 bounded unit proofs | PASS |
| R1-R1 projection recovery unit | PASS |
| D1 durable conformance unit | PASS |
| R1 Mongo hot-path (E=5000, P=25, analyzer 100%) | PASS |
| R1-R2 Mongo process recovery | PASS |

## Files changed

- `intergrax/runtime/diagnostics/functional_evidence_append_intent.py` (new)
- `intergrax/runtime/diagnostics/functional_evidence_projection_repairer.py` (new)
- `intergrax/runtime/diagnostics/document_store_functional_evidence_persistence.py`
- `tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r2_append_crash_safety.py` (new)
- `tests/system/functional_diagnostics_read_r1_r2/` (new)
- `docs/project/architecture/DIAGNOSTICS.md`
- `docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_R1_QUALIFICATION.md`

## Final architecture statement

```text
CANONICAL FUNCTIONAL EVIDENCE = SOLE SOURCE OF TRUTH
UNFINISHED FUNCTIONAL EVIDENCE APPEND = DURABLY DETECTABLE (appendpending)
DERIVED PROJECTION = CRASH-REPAIRABLE FROM CANONICAL
QUERY FAST PATH = MANIFEST COMPLETE AND ZERO UNRESOLVED APPEND INTENTS
SILENT CANONICAL EVIDENCE OMISSION = IMPOSSIBLE WITHIN QUALIFIED FAILURE MODEL
```

## Independent audit finding (R1-R3 supersession)

R1-R2 allowed reader to classify `pending + canonical missing` as orphan and delete the intent. Under concurrent writer/reader timing this could delete an **active** writer intent, then allow canonical write after reader — yielding `canonical` without repaired projections and without pending guard → **silent canonical omission** risk.

**R1-R3 closure:** see [`DIAG_FUNCTIONAL_READ_R1_R3_QUALIFICATION.md`](DIAG_FUNCTIONAL_READ_R1_R3_QUALIFICATION.md).
