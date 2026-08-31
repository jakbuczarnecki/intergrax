# Diagnostic enterprise scale qualification matrix

Enterprise-scale qualification ledger for diagnostics beyond functional HARDEN proofs.

Status vocabulary:

- `PROVEN` — bounded proof exists in this repository for the stated semantics
- `REQUALIFICATION_REQUIRED` — prior proof invalidated; remediation in flight
- `NOT_YET_QUALIFIED` — not proven in-repo yet; do not treat as production scale guarantee

## E1 — Scalable Problem reads (`DIAG-ENTERPRISE-1` / `DIAG-ENTERPRISE-1-R1` / `DIAG-ENTERPRISE-1-R2` / `DIAG-ENTERPRISE-1-R3` / `DIAG-ENTERPRISE-1-R4` / `DIAG-ENTERPRISE-1-R5` / `DIAG-ENTERPRISE-1-R6`)

**Status:** `PROVEN` — R6 first-page failure recovery (awaiting E1 freeze decision)

Operator Problem list reads are bounded by page/query instead of materializing entire tenant cardinality. Stale/orphan derived list projections have a bounded maintenance path; active writer transitions are never deleted by maintenance. Callers cannot bypass `MIN_SAFE_PROJECTION_AGE` (5 minutes).

| Capability | Semantics | Proof |
|---|---|---|
| Bounded persistence query | `ProblemPersistence.query_problems(tenant_id, status?, limit, cursor?) → ProblemListPage` | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_scalable_problem_reads.py` |
| Concurrent transition tolerance | index leads / canonical leads → skip, not false `IntegrityError` | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r1_read_index_races.py` |
| Projection reconciliation | `reconcile_list_indexes(minimum_projection_age, …)` bounded; proven orphan delete / proven stale repair | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r2_list_index_reconciliation.py` |
| Maintenance safety age | `MIN_SAFE_PROJECTION_AGE`; caller cannot request zero/unsafe age; future projections safe | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r3_maintenance_safety.py` |
| Recoverable projection health | cumulative telemetry preserved; current health recovers after full clean cycle on same `(tenant_id, scope)` | R3 + R4 suites |
| Maintenance cycle identity | per `(tenant_id, scope)` process-local cycle state; abandoned/incomplete cycles cannot be masked by unrelated clean scans; restart clears health state | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r4_maintenance_cycle_identity.py` |
| Maintenance single-flight | one active page per `(tenant_id, scope)`; same-key parallel continuation rejected; different tenant/scope concurrent; ownership released on exception | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r5_single_flight.py` |
| First-page failure recovery | newly started cycle first-page failure rolls back only process-local cycle state (restore prior degraded snapshot or remove fresh entry); continuation failure retains cycle; persistence/telemetry not rolled back | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_r6_first_page_failure_recovery.py` |
| Projection telemetry | skip/repair counters + `HEALTHY`/`DEGRADED` health snapshot | R2 + R3 + R4 suites |
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

## E2 — Bounded occurrence history (`DIAG-ENTERPRISE-2`)

**Status:** `IN_PROGRESS` — core abstraction landed; full proof matrix incomplete

Canonical `Problem` is a bounded aggregate (no inline `occurrences` / `current_subject_refs`). Durable occurrence history uses `ProblemOccurrencePersistence` with `DocumentStoreProblemOccurrencePersistence` over `ConditionalDocumentStore` (InMemory + Mongo-capable).

| Capability | Semantics | Proof |
|---|---|---|
| Bounded Problem aggregate | no unbounded occurrence tuple on `Problem` | `tests/unit/runtime/diagnostics/test_diag_enterprise_2_occurrence_persistence.py` |
| Occurrence persistence contract | `append_if_absent`, `query_occurrences`, `aggregate_stats` | same + conformance in progress |
| Lifecycle write protocol | occurrence append → aggregate converge | `test_problem_lifecycle.py` |
| Paginated occurrence read | `DiagnosticReadService.list_problem_occurrences` | `test_diagnostic_read_service.py` |
| 100k bounded proof | aggregate shape stable; page limit=100 | `test_100k_bounded_aggregate_and_paginated_history` |
| Legacy v1 migration | `problem_occurrence_migration.py` | not yet fully proven |
| Mongo durability proof | restart + pagination | not yet run in this iteration |
| E1 regression | R1–R6 | pending full green suite |

**Design notes**

- Occurrence partition: `intergrax.diagnostic_problem_occurrence.v1:{tenant_id}:{problem_id}`
- Row key: `occ:{inverted_observed_at_micros}:{occurrence_id}` where `occurrence_id = subject_ref.index_token`
- Subject ownership index remains on `ProblemPersistence` via `indexed_subject_refs` on create/update (not on aggregate)

## E3 — Contention / hot-partition load

**Status:** `NOT_YET_QUALIFIED`

Tenant partition hotspot and write-contention qualification remain separate from read-index delivery.

## E4 — Async distributed P4

**Status:** `NOT_YET_QUALIFIED`

Distributed async diagnostic platform proofs are not part of enterprise read-index slice.
