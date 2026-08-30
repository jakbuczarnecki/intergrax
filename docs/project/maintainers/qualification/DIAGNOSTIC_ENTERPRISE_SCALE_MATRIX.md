# Diagnostic enterprise scale qualification matrix

Enterprise-scale qualification ledger for diagnostics beyond functional HARDEN proofs.

Status vocabulary:

- `PROVEN` — bounded proof exists in this repository for the stated semantics
- `NOT_YET_QUALIFIED` — not proven in-repo yet; do not treat as production scale guarantee

## E1 — Scalable Problem reads (`DIAG-ENTERPRISE-1`)

**Status:** `PROVEN`

Operator Problem list reads are bounded by page/query instead of materializing entire tenant cardinality.

| Capability | Semantics | Proof |
|---|---|---|
| Bounded persistence query | `ProblemPersistence.query_problems(tenant_id, status?, limit, cursor?) → ProblemListPage` | `tests/unit/runtime/diagnostics/test_diag_enterprise_1_scalable_problem_reads.py` |
| Public ordering | `last_seen_at DESC`, `problem_id ASC` tie-break | same + `test_diagnostic_read_service.py` |
| Status filter without full scan | `OPEN` / `RESOLVED` / all via derived list index scopes | same |
| Opaque query-bound cursor | HMAC envelope binds tenant + status filter + store continuation | same |
| Integrity fail-closed | orphan/stale list index → `ProblemPersistenceIntegrityError` | same |
| Real Mongo provider | paginated query on DocumentStore/Mongo path | `tests/integration/runtime/test_diag_enterprise_1_mongo_scalable_read.py` |
| 10k bounded-work proof | one index query page + ≤ page-size canonical fetches | `test_bounded_query_does_not_materialize_full_tenant` |

**Design notes**

- Canonical Problem records remain truth; `list:{scope}:…` rows are **derived read indexes** only.
- `DiagnosticReadService.list_problems` uses persistence query; it does **not** call full `list_for_tenant`.
- `total_count` is exact only when `cursor is None` and `has_more is False`; otherwise `None` (no full-scan count).
- Pagination is cursor-based continuation, not snapshot isolation; concurrent updates may shift ordering between pages without loops or cross-tenant leakage.

## E2 — Bounded occurrence history

**Status:** `NOT_YET_QUALIFIED`

`Problem.occurrences` remains embedded in canonical records; unbounded occurrence history is out of scope for E1.

## E3 — Contention / hot-partition load

**Status:** `NOT_YET_QUALIFIED`

Tenant partition hotspot and write-contention qualification remain separate from read-index delivery.

## E4 — Async distributed P4

**Status:** `NOT_YET_QUALIFIED`

Distributed async diagnostic platform proofs are not part of enterprise read-index slice.
