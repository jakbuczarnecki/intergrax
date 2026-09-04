# DIAG-FUNCTIONAL-SCALE-S1 Qualification

**Verdict:** PASS (canonical STANDARD envelope)

**Date:** 2026-09-03

**Branch:** `development`

**Qualified SHA:** `98caff3af51b2951b8f0704ac7f96fea526cbfd5`

## History

1. **Harness implementation** — reusable typed scale qualification layer (`FunctionalDiagnosticsScaleRunner`, `ScaleBackendProbe`, workload generator, manifest oracle, gates S1-A … S1-O).
2. **SMOKE preflight** — dev envelope PASS after harness integration fixes (isolated S1-B collection, deterministic workload IDs, writer-batch gate semantics).
3. **First canonical STANDARD run** — real MongoDB, frozen profile, D1 contract regression PASS — **PASS** (2026-09-03).

> Canonical run command: `uv run python -m tests.system.functional_diagnostics_scale.runner`
>
> Machine artifacts: `.tmp/session/diag-functional-scale-s1/`

## Scale claim (envelope-bounded)

```text
FUNCTIONAL DIAGNOSTICS PRODUCTION SCALE
= QUALIFIED FOR THE S1 TESTED ENVELOPE
```

**Tested envelope (STANDARD profile, seed `20260903`):**

| Parameter | Value |
| --------- | ----- |
| tenants | 12 |
| executions / tenant | 120 (1,440 total) |
| typical evidence / execution | 18 |
| heavy executions / tenant | 2 |
| heavy evidence / execution | 240 |
| total evidence | 31,032 |
| writer processes | 4 |
| reader processes | 4 |
| page size | 25 |
| document-store query page limit | 100 |
| analyzer sample executions / tenant | 2 |
| scale-curve probe evidence / execution | 20 |

## Architecture scale audit

| Component | Operation | Complexity | Bounded by | Potential scale risk |
| --------- | --------- | ---------- | ---------- | -------------------- |
| `FunctionalEvidenceRecorder.append` | delegate to persistence | O(1) call | contract | none in core |
| `DocumentStoreFunctionalEvidencePersistence.append` | `put_if_absent` canonical + index | O(1) Mongo ops / evidence | partition + row key | duplicate index repair on retry |
| `DocumentStoreFunctionalEvidencePersistence.query_evidence` | prefix query execution index + canonical gets | O(E) index rows + O(E) canonical gets | execution scope (task+run prefix) | **N+1 canonical reads**; full execution materialized in memory before pagination |
| `FunctionalDiagnosticAnalyzer.analyze` | paginated `query_evidence` per check/kind | O(checks × pages) | page_size + execution scope | repeated scans per evidence kind |
| Mongo `_MongoDBDocumentStore.query` | partition + row_key regex prefix | O(log N + k) with index | partition_key + row_key prefix | regex prefix on row_key uses compound index |
| Execution index rows | `exec:{task}:{run}:{evidence_id}` | O(1) insert | execution cardinality | none |
| Canonical rows | `record:{evidence_id}` | O(1) insert | tenant partition | none |

### Read-path behavior (explicit)

- **Tenant scan?** No — queries use `partition_key = intergrax.functional_evidence.v1:{tenant_id}`.
- **Collection scan?** No for execution-scoped reads — `row_key_prefix = exec:{task}:{run}:` with partition filter.
- **Execution materialization?** Yes — `_sorted_entries_for_execution` + `_records_for_execution` load **all** execution evidence before pagination filtering.
- **N+1?** Yes — one canonical `get` per index row during index scan, repeated in `_records_for_execution` (≈ **2×O(E)** backend gets per query page build).
- **Pagination strategy?** Keyset cursor over sorted execution entries; late arrivals before consumed cursor may require subsequent reconstruction cycle (contract in `FunctionalEvidencePersistence.query_evidence`).

### Mongo production indexes

Only production index used (created by `create_mongodb_document_store` → `_ensure_document_key_index`):

```text
uq_intergrax_document_key: { partition_key: 1, row_key: 1 } UNIQUE
```

S1-L observed `documents_examined=1` for execution-prefix query vs `total_docs=62,068` — index prefix scan, not collection scan.

## Generic scale architecture

```text
FunctionalDiagnosticsScaleRunner (tests/system/functional_diagnostics_scale/runner.py)
        ↓
ScaleBackendProbe (Protocol)
        ↑
MongoFunctionalDiagnosticsScaleProbe  |  SyntheticFunctionalDiagnosticsScaleProbe (S1-N)
```

- Explicit composition / dependency injection — no dynamic registry.
- Mongo-specific code only in `mongodb_backend.py` qualification plugin.
- `intergrax/runtime/diagnostics/*` — zero Mongo imports.

## S1 gate matrix (canonical STANDARD)

| Gate | Result | Summary |
| ---- | ------ | ------- |
| S1-A High-cardinality append | PASS | 31,032 expected = actual, 0 duplicates |
| S1-B Query boundedness | PASS | sub-linear latency growth (203→734→422 ms) |
| S1-C Pagination completeness | PASS | heavy execution 240/240 across 10 pages |
| S1-D Multi-tenant isolation | PASS | tenant_leakage=0 |
| S1-E Concurrent writers | PASS | 4 workers, 0 errors |
| S1-F Concurrent readers | PASS | 4 workers, 0 errors |
| S1-G Concurrent read/write | PASS | concurrent read+write OK |
| S1-H Idempotency contention | PASS | 2 workers, 0 errors |
| S1-I Conflict contention | PASS | 20 conflicts detected |
| S1-J Analyzer fidelity | PASS | 24 samples, 0 mismatches (100%) |
| S1-K Resource boundedness | PASS | max scoped read 27,890 ms < 60,000 ms gate |
| S1-L Mongo/index efficiency | PASS | examined=1, total_docs=62,068 |
| S1-M Recovery after load | PASS | heavy execution 240/240 recovered |
| S1-N Backend pluginability | PASS | synthetic probe without runner rewrite |
| S1-O Delivery decoupling | PASS | static audit — no delivery coupling in diagnostics core |

## S1-B scale curve (isolated collection)

| Dataset | Total DB evidence | Probe execution evidence | Read latency |
| ------- | ----------------- | ------------------------ | ------------ |
| small | 1,620 | 20 | 203 ms |
| medium | 11,220 | 20 | 734 ms |
| large | 40,020 | 20 | 422 ms |

Growth is **not** approximately linear with total collection cardinality.

## Latency (canonical STANDARD)

| Operation | p50 | p95 | p99 | max |
| --------- | --- | --- | --- | --- |
| append | 31.0 ms | 63.0 ms | 94.0 ms | 360.0 ms |
| execution read | 234.0 ms | 437.0 ms | 29,343.0 ms | 38,531.0 ms |
| analyze | 297.0 ms | 391.0 ms | 453.0 ms | 453.0 ms |

## Throughput

| Metric | Value |
| ------ | ----- |
| writes/sec | 11.07 |
| reads/sec | (descriptive — 1,440 execution reads in reader phase) |
| analyses/sec | (descriptive — 24 analyzer samples) |

## Correctness (mandatory = 0)

```text
lost_evidence              = 0
unexpected_duplicates      = 0
tenant_leakage             = 0
task_leakage               = 0
run_leakage                = 0
integrity_errors           = 0
unexpected_errors          = 0
timeouts                   = 0
assessment_mismatch        = 0
```

## Hardware / environment

| Field | Value |
| ----- | ----- |
| CPU cores | 20 |
| Mongo deployment | local (`mongodb://localhost:27017`) |
| Mongo documents (main collection) | 62,068 |
| Mongo storage | ~8.7 MB |
| RSS probe | unavailable on Windows harness (null) |

## Real backend

| Field | Value |
| ----- | ----- |
| provider | `mongodb` |
| DocumentStore | `_MongoDBDocumentStore` |
| database | `intergrax_diag_scale_s1` |
| collection | `intergrax_diag_scale_s1_<run-uuid>` |
| production factory | `create_mongodb_document_store()` |

## S1-O Delivery decoupling readiness

Static audit of `intergrax/runtime/diagnostics/` and `intergrax/runtime/observability/` found **no** `TaskQueue`, `queue_mode`, or `WorkSubmission` coupling.

Frozen future model (S2):

```text
WorkSubmissionStrategy
    ↑
    ├── DirectWorkSubmissionStrategy
    └── QueuedWorkSubmissionStrategy

Queued → TaskQueue → provider plugins

Authority:
  TaskQueue         = delivery state
  RuntimeEvent      = execution truth
  FunctionalEvidence = functional truth
```

## D1 regression

D1 contract qualification PASS before canonical run (`run_contract_qualification()`).

## Analyzer

`FunctionalDiagnosticAnalyzer` — **UNCHANGED**.

## What S1 proves

```text
FUNCTIONAL DIAGNOSTICS PRODUCTION SCALE
= QUALIFIED FOR THE S1 TESTED ENVELOPE
```

Bounded execution-scoped reads, multi-tenant isolation, concurrent read/write safety, pagination completeness, analyzer fidelity, process recovery, and real MongoDB durability under the STANDARD envelope above.

## What S1 does NOT prove

- queue-mediated scale (S2)
- distributed worker scale (S3)
- larger envelopes / multi-region / DR
- other durable providers
- H1

## Next roadmap

```text
Functional Diagnostics foundation     ✅
Q1–Q5 universality                    ✅
D1 durability/recovery                ✅
S1 core scale                         ✅

S2 Pluggable delivery / queues        ▶ NEXT
S3 Distributed worker scale           ⏳
S4 Documentation closure              ⏳
H1                                    ⏳
DR                                    ⏳
```

## Machine artifacts

- `.tmp/session/diag-functional-scale-s1/scale-profile.json` — frozen before run
- `.tmp/session/diag-functional-scale-s1/qualification-report.json`
- `.tmp/session/diag-functional-scale-s1/latency-metrics.json`
- `.tmp/session/diag-functional-scale-s1/resource-metrics.json`
- `.tmp/session/diag-functional-scale-s1/run.log`

## Post-S1 read-path hardening (DIAG-FUNCTIONAL-READ-R1)

S1 PASS at `98caff3af51b2951b8f0704ac7f96fea526cbfd5` exposed a **production limitation** in the pre-R1 read path: `query_evidence()` materialized the full execution (≈2×O(E) canonical reads per page). S1 historical PASS is **not** invalidated — it documents the qualified envelope at that SHA.

R1 remediation (order-aware `execidx:` v2 derived projection, bounded incremental scan) is qualified separately: [`DIAG_FUNCTIONAL_READ_R1_QUALIFICATION.md`](DIAG_FUNCTIONAL_READ_R1_QUALIFICATION.md).
