# Diagnostic enterprise scale qualification matrix

Enterprise-scale qualification ledger for diagnostics beyond functional HARDEN proofs.

Status vocabulary:

- `PROVEN` — bounded proof exists in this repository for the stated semantics
- `REQUALIFICATION_REQUIRED` — prior proof invalidated; remediation in flight
- `NOT_YET_QUALIFIED` — not proven in-repo yet; do not treat as production scale guarantee

## E1 — Scalable Problem reads (`DIAG-ENTERPRISE-1` / `DIAG-ENTERPRISE-1-R1` / `DIAG-ENTERPRISE-1-R2` / `DIAG-ENTERPRISE-1-R3`)

**Status:** `PROVEN` — **FINAL PROVEN** (R3 safety-age + recoverable health)

Operator Problem list reads are bounded by page/query instead of materializing entire tenant cardinality. Stale/orphan derived list projections have a bounded maintenance path; active writer transitions are never deleted by maintenance. Callers cannot bypass `MIN_SAFE_PROJECTION_AGE` (5 minutes).

| Capability | Semantics | Proof |
|---|---|---|
| Bounded persistence query | `ProblemPersistence.query_problems(tenant_id, status?, limit, cursor?) → ProblemListPage` | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_scalable_problem_reads.py` |
| Concurrent transition tolerance | index leads / canonical leads → skip, not false `IntegrityError` | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r1_read_index_races.py` |
| Projection reconciliation | `reconcile_list_indexes(minimum_projection_age, …)` bounded; proven orphan delete / proven stale repair | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r2_list_index_reconciliation.py` |
| Maintenance safety age | `MIN_SAFE_PROJECTION_AGE`; caller cannot request zero/unsafe age; future projections safe | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r3_maintenance_safety.py` |
| Recoverable projection health | cumulative telemetry preserved; current health recovers after full clean cycle | same |
| Projection telemetry | skip/repair counters + `HEALTHY`/`DEGRADED` health snapshot | R2 + R3 suites |
| Large garbage proof | 10k orphan/stale indexes + 1k valid Problems; bounded maintenance pages | `test_large_garbage_reconciliation_and_read_recovery` |
| Public ordering | `last_seen_at DESC`, `problem_id ASC` tie-break | same + `test_diagnostic_read_service.py` |
| Status filter without full scan | `OPEN` / `RESOLVED` / all via derived list index scopes | same |
| Authenticated query-bound cursor | HMAC envelope binds tenant + status filter + store continuation; production secret ≥32 bytes | R1/R2 cursor suites |
| Integrity fail-closed | same-version metadata mismatch / id mismatch → `ProblemPersistenceIntegrityError` | same |
| Real Mongo provider | paginated query on DocumentStore/Mongo path | `tests/integration/runtime/test_diag_enterprise_1_mongo_scalable_read.py` |
| Real Mongo reconciliation | proven orphan delete + query recovery | `tests/integration/runtime/test_diag_enterprise_1_r2_mongo_reconciliation.py` |
| 10k bounded-work proof | bounded index examination + canonical fetches | `test_bounded_query_does_not_materialize_full_tenant` |
| No full-tenant list API | `list_for_tenant` removed from production contract | architecture + conformance helpers |

**Design notes**

- Canonical Problem records remain truth; `list:{scope}:…` rows are **derived read indexes** with `record_version` and `projection_written_at` (v2).
- `DiagnosticReadService.list_problems` uses persistence query only; maintenance is explicit (`reconcile_list_indexes`), not hot-path destructive.
- `total_count` is exact only when `cursor is None` and `has_more is False`; dashboard exposes `problem_count_is_exact` / `open_problem_count_is_exact`.
- Pagination is cursor-based continuation, not snapshot isolation; concurrent updates may shift ordering between pages without loops or cross-tenant leakage.
- Production cursor secret: `INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET` — minimum 32 UTF-8 bytes, randomly generated; restart invalidates prior cursors.

## E2 — Bounded occurrence history

**Status:** `NOT_YET_QUALIFIED`

`Problem.occurrences` remains embedded in canonical records; unbounded occurrence history is out of scope for E1.

## E3 — Contention / hot-partition load

**Status:** `NOT_YET_QUALIFIED`

Tenant partition hotspot and write-contention qualification remain separate from read-index delivery.

## E4 — Async distributed P4

**Status:** `NOT_YET_QUALIFIED`

Distributed async diagnostic platform proofs are not part of enterprise read-index slice.
