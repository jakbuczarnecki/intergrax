# DIAG-FUNCTIONAL-SCALE-S1 Qualification

**Verdict:** see machine artifact (`.tmp/session/diag-functional-scale-s1/qualification-report.json`)

**Date:** 2026-09-03

**Branch:** `development`

## Purpose

S1 qualifies functional diagnostics under realistic high-cardinality load:

- bounded read/write model
- multi-tenant isolation
- concurrent writers/readers (multi-process)
- pagination completeness
- analyzer fidelity under scale
- Mongo index/prefix query behavior

S1 is **not** a hyperscale speed benchmark. Qualification is always scoped to the frozen profile envelope.

## Architecture audit (hot path)

| Layer | Operation | Complexity | Bounded? | Risk |
| ----- | --------- | ----------: | -------: | ---- |
| append canonical | `put_if_absent(partition, record:evidence_id)` | O(1) index lookup | yes (per record) | low |
| append index | `put_if_absent(partition, exec:task:run:evidence_id)` | O(1) | yes (per record) | low |
| execution index query | `query(partition, row_key_prefix=exec:task:run:)` paginated | O(execution index pages) | yes (execution prefix) | medium at very heavy execution |
| canonical record resolution | `get` per index entry | O(execution evidence) | bounded by execution | high fan-out for evidence-heavy execution |
| page cursor | sort execution entries + keyset cursor | O(execution evidence) materialized | bounded by execution, not tenant | memory within execution |
| analyzer reconstruction | paginated `query_evidence` until exhausted | O(execution evidence) | yes (execution scope) | late-arrival requires new cycle per contract |

**Core invariant under test:** reading one execution must not require scanning all tenant/system evidence. Mongo path uses `partition_key` + `row_key` prefix (compound unique index).

**Late arrival contract (existing):** monotonic ordered scan; evidence inserted before consumed cursor may require subsequent reconstruction cycle.

## Canonical runner

```bash
uv run python -m tests.system.functional_diagnostics_scale.runner
```

Profiles: `SMOKE`, `STANDARD` (canonical S1), `STRESS`.

If Mongo/pymongo unavailable: **BLOCKED** (no in-memory fallback).

## Production provider path

```text
create_mongodb_document_store()
  → _MongoDBDocumentStore
  → DocumentStoreFunctionalEvidencePersistence
```

Qualification collection: `intergrax_diag_scale_s1_<uuid>` in database `intergrax_diag_scale_s1`.

## Gates

| Gate | Description |
| ---- | ----------- |
| S1-A | High-cardinality append correctness |
| S1-B | Execution query boundedness curve (small/medium/large total cardinality) |
| S1-C | Pagination completeness (heavy execution, multi-page) |
| S1-D | Multi-tenant isolation |
| S1-E | Concurrent writers (multi-process) |
| S1-F | Concurrent readers |
| S1-G | Concurrent read/write |
| S1-H | Idempotency under contention |
| S1-I | Conflict under contention |
| S1-J | Analyzer fidelity (deterministic replay fingerprint) |
| S1-K | Scoped read resource boundedness |
| S1-L | Mongo index/query efficiency (provider plugin) |
| S1-M | Recovery after load (fresh adapter) |
| S1-N | Backend pluginability (synthetic probe) |

## Artifacts

```text
.tmp/session/diag-functional-scale-s1/
  scale-profile.json
  qualification-report.json
  latency-metrics.json
  resource-metrics.json
  scale-manifest.json
  run.log
```

## If FAILED

Honest baseline preserved. Remediation tracked as **S1-R1** — do not retune profile/thresholds in the same task.
