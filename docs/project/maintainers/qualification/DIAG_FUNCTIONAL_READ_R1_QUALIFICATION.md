# DIAG-FUNCTIONAL-READ-R1 Qualification

**Verdict:** PASS

**Date:** 2026-09-03

**Branch:** `development`

**Qualified SHA (S1 envelope):** `98caff3af51b2951b8f0704ac7f96fea526cbfd5`

**Implementation HEAD:** see git `development` after R1 merge

## Baseline problem (pre-R1)

`DocumentStoreFunctionalEvidencePersistence.query_evidence()` materialized the full execution before pagination:

1. load all execution index rows (`exec:{task}:{run}:{evidence_id}`)
2. canonical `get()` per index row during sort preparation
3. sort/materialize all entries in memory
4. canonical `get()` again for every entry into `records_by_id`
5. slice page after full execution build

Single-page cost ≈ **2 × O(E)** backend canonical reads when **E >> page_size**.

## Architecture decision

| Option | Summary | Verdict |
| ------ | ------- | ------- |
| A — order-aware execution index key | Derived `execidx:` row keys encode `(recorded_at, evidence_id)` ordering; bounded prefix scan | **Selected** |
| B — DocumentStore data sort/filter only | Relies on provider sort capability without order in key | Rejected — weaker portability for keyset resume |
| C — bounded scan + page-only canonical resolution without order key | Still requires full scan/sort for keyset | Rejected — does not fix hot path |

**Selected:** Option A with v2 derived projection (`intergrax.functional_evidence.index.v2`), separate `execidx:` prefix from legacy `exec:` v1 rows. Canonical `record:{evidence_id}` remains sole truth.

## Contract changes

- No change to `FunctionalEvidencePersistence` public semantics.
- Append path now writes **v1 + v2** derived indexes (both projections; canonical unchanged).
- Query path uses **v2 only** after optional one-time v1→v2 rebuild.

## Index model

| Schema | Row key | Metadata |
| ------ | ------- | -------- |
| v1 (legacy) | `exec:{task}:{run}:{evidence_id}` | `evidence_id` reference only |
| v2 (query) | `execidx:{task}:{run}:{micros:020d}:{evidence_id}` | `evidence_id`, `recorded_at`, `kind`, optional `attempt_id` |

Ordering: ascending `(recorded_at, evidence_id)` via zero-padded epoch micros + evidence_id tie-break.

## Query algorithm (bounded)

```text
ensure v2 projection (lazy rebuild from v1 if needed)
→ incremental execidx prefix query (DocumentStore cursor)
→ filter kind/attempt from index metadata (no canonical get)
→ canonical get only for page candidates
→ integrity validate index metadata vs canonical
→ lookahead for next_cursor (incremental, not full execution)
```

## Operation-count proof

| Scenario | E | P | index queries | canonical gets (first page) |
| -------- | - | - | ------------- | --------------------------- |
| Old path (S1 audit) | 1000+ | 25 | all pages | ≈ 2×E |
| In-memory unit gate | 1000 | 25 | ≤ 4 | ≤ 50 (`<< E`) |
| Real Mongo R1 profile | 5000 | 25 | 3 | **25** |

## Real Mongo result (focused R1 profile)

Command: `uv run python -m tests.system.functional_diagnostics_read_r1.mongo_qualification`

Artifacts: `.tmp/session/diag-functional-read-r1/mongo-qualification.json`

| Probe | Latency (ms) |
| ----- | ------------ |
| first page | 151.4 |
| middle page | 95.2 |
| final page | 86.3 |
| filtered page | 182.5 |

Union pagination over 5000 evidence: **5000/5000** (100% fidelity).

Compare S1 baseline execution read (same envelope family, pre-R1): p50 234 ms · p95 437 ms · p99 29,343 ms · max 38,531 ms — not apples-to-apples (different profile/hardware) but structural gate is **canonical reads no longer scale with E for first page**.

## Compatibility / projection upgrade

- Existing v1-only persisted evidence: `FunctionalEvidenceIndexRebuilder.ensure_v2_projection()` on first query (idempotent).
- Canonical rows unchanged; no dual truth.
- Orphan/mismatch: fail closed (`FunctionalEvidencePersistenceIntegrityError`).

## Regression summary

| Gate | Result |
| ---- | ------ |
| Unit bounded-read proofs | PASS |
| Durable persistence conformance | PASS |
| DIAG-FUNCTIONAL-1/2 hardening | PASS |
| Q1–Q4 evidence fidelity | PASS (25 tests) |
| D1 contract (unit matrix) | PASS |
| Mongo focused R1 | PASS |

## Final architecture statement

```text
FUNCTIONAL EVIDENCE READ PATH
= BOUNDED BY PAGE / INCREMENTAL SCAN

CANONICAL FUNCTIONAL EVIDENCE
= UNCHANGED SOURCE OF TRUTH

DIAGNOSTIC STORAGE ACCELERATION
= PROVIDER-NEUTRAL / CAPABILITY-BASED
```

## Production files changed

- `intergrax/runtime/diagnostics/functional_evidence_execution_index.py`
- `intergrax/runtime/diagnostics/functional_evidence_index_rebuilder.py`
- `intergrax/runtime/diagnostics/document_store_functional_evidence_persistence.py`
- `tests/unit/runtime/diagnostics/test_diag_functional_read_r1_bounded_reads.py`
- `tests/system/functional_diagnostics_read_r1/mongo_qualification.py`
- `docs/project/architecture/DIAGNOSTICS.md`
- `docs/project/maintainers/qualification/DIAG_FUNCTIONAL_SCALE_S1_QUALIFICATION.md` (post-S1 note)

## Independent audit finding (R1-R1)

Pre-R1-R1 `ensure_v2_projection()` treated **any** v2 row as proof of complete projection (`query v2 limit=1 → skip rebuild`). Crash mid-rebuild left partial v2 navigable → silent evidence loss.

**R1-R1 closure:** see [`DIAG_FUNCTIONAL_READ_R1_R1_QUALIFICATION.md`](DIAG_FUNCTIONAL_READ_R1_R1_QUALIFICATION.md). R1 PASS above remains valid for bounded-read architecture; crash-safe projection completeness is qualified only after R1-R1 final SHA.
