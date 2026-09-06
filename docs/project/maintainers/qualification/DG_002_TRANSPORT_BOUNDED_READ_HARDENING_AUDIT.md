# DG-002 Transport Bounded Read Hardening — Audit (R1)

**Verdict:** PASS

**Date:** 2026-09-06

**Branch:** `development`

**Start HEAD:** `0b5d356314da6793c1f953981fd60473c5dc35ab`

**Task:** `DG-002-TRANSPORT-BOUNDED-READ-HARDENING-AUDIT-R1` — audit / design only; no production code changes.

---

## Executive summary

`CausalEvidencePersistence.list_for_transport_task()` and `list_for_execution()` return unbounded `tuple[PlatformCausalEvidence, ...]`. `DocumentStoreCausalEvidencePersistence` paginates internally but materializes, sorts, and returns the full result set. **Internal pagination ≠ bounded caller read.**

Enterprise closure for DG-002 transport scope discovery requires a new **paged canonical API** on `CausalEvidencePersistence`, retention of existing list methods as compatibility facades, and migration of `CausalTransportScopeProvider` to incremental `page_for_transport_task()` consumption.

**Ordering hardening is required:** current index row keys end with `evidence_id` only; DocumentStore cursor order is row-key order, not canonical `(recorded_at, evidence_id)`.

---

## Confirmed gap

| Finding | Status |
| ------- | ------ |
| DG-002 TRANSPORT BOUNDED READ GAP | **CONFIRMED** |
| Root cause | Unbounded tuple public API |
| DocumentStore lower-level pagination available | **YES** (`query` → `DocumentQueryPageV1`) |
| Existing canonical causal order | `(recorded_at, evidence_id)` via `causal_evidence_query_order_key` |
| Current index supports canonical ordered paging directly | **NO** |
| Ordering / index hardening required | **YES** |

---

## Audited artifacts

| Path | Role |
| ---- | ---- |
| `intergrax/runtime/observability/causal_evidence_persistence.py` | Abstract contract |
| `intergrax/runtime/observability/memory_causal_evidence_persistence.py` | In-memory backend |
| `intergrax/runtime/observability/document_store_causal_evidence_persistence.py` | Durable backend |
| `intergrax/runtime/observability/persistence_conformance.py` | Conformance harness |
| `intergrax/integrations/contracts/document_store.py` | Lower-level paging primitive |
| `intergrax/runtime/diagnostics/providers/causal_transport_scope_provider.py` | Transport discovery (DG-002 blocker consumer) |
| `intergrax/runtime/diagnostics/execution_reconstruction.py` | Execution reconstruction (separate scale path) |

### Implementation inventory

Only two `CausalEvidencePersistence` implementations exist:

1. `InMemoryCausalEvidencePersistence` — tests, conformance, local lab
2. `DocumentStoreCausalEvidencePersistence` — production durable path

No additional backends discovered.

---

## Current unbounded behavior (evidence)

### Public contract

```python
def list_for_transport_task(...) -> tuple[PlatformCausalEvidence, ...]
def list_for_execution(...) -> tuple[PlatformCausalEvidence, ...]
```

No `limit`, no `cursor`, no page type.

### DocumentStore backend (`_list_indexed`)

1. Loop: `query(partition, limit=5000, row_key_prefix=..., cursor=...)`
2. Append all `page.documents`
3. For **every** index row: fetch canonical record, validate, decode
4. `decoded.sort(key=causal_evidence_query_order_key)`
5. Return full tuple

**INTERNAL PAGINATION ≠ BOUNDED CALLER READ.**

### In-memory backend

`list_for_*` resolves all evidence IDs for the key, sorts all matching records, returns full tuple. No paging surface.

### Transport provider

`CausalTransportScopeProvider._list_transport_evidence` calls `list_for_transport_task()` once and materializes the full history before scope classification. `candidate_limit` bounds only the public candidate projection in `_classify_execution_scopes`; it does **not** bound persistence read.

---

## Caller classification

| Caller | Kind | API used | DG-002 action |
| ------ | ---- | -------- | ------------- |
| `CausalTransportScopeProvider` | **PRODUCTION** | `list_for_transport_task` | **MUST migrate** to `page_for_transport_task` |
| `ExecutionReconstructor` | **PRODUCTION** | `list_for_execution` | **Keep** `list_for_execution` (DG-004 / scale; out of DG-002 migration scope) |
| `persistence_conformance.py` | **CONFORMANCE** | both list APIs | Extend for page APIs; keep list conformance |
| `test_durable_causal_evidence_persistence.py` | TEST | both | Keep + add page tests |
| `test_background_causal_evidence_admission_paths.py` | TEST | transport list | Keep |
| `test_required_audit_evidence_admission.py` | TEST | execution list | Keep |
| `test_causal_transport_scope_provider.py` | TEST | transport list (mocks) | Update when provider migrates |
| `test_execution_reconstruction.py` | TEST | execution list override | Keep |

No other production callers of `list_for_transport_task` or `list_for_execution` found under `intergrax/`.

---

## Frozen enterprise contract (design)

### Page model

```python
@dataclass(frozen=True, slots=True)
class CausalEvidencePage:
    items: tuple[PlatformCausalEvidence, ...]
    next_cursor: str | None
```

Semantics: one bounded page of causal evidence in **canonical query order** `(recorded_at ASC, evidence_id ASC)`. No generic dict page.

### Canonical paged methods (new)

```python
def page_for_execution(
    self,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    limit: int,
    cursor: str | None = None,
) -> CausalEvidencePage

def page_for_transport_task(
    self,
    *,
    tenant_id: str,
    provider: str,
    transport_task_id: str,
    limit: int,
    cursor: str | None = None,
) -> CausalEvidencePage
```

Both axes use the same `CausalEvidencePage` abstraction (symmetry with existing execution + transport list APIs).

### Compatibility facades (retained)

```python
def list_for_transport_task(...):
    return _consume_all_pages(page_for_transport_task(...))

def list_for_execution(...):
    return _consume_all_pages(page_for_execution(...))
```

| API | Role | Enterprise bounded |
| --- | ---- | ------------------ |
| `page_for_*` | Canonical bounded read | **YES** |
| `list_for_*` | Compatibility / convenience | **NO** |

Do not add new production consumers of `list_for_*`. Static enforcement recommended: diagnostics must not call `list_for_transport_task`.

### Page limit validation

Reusable validator: `1 <= limit <= CAUSAL_EVIDENCE_QUERY_MAX_LIMIT`.

- No `bool` accepted as limit.
- No `limit=None` (unbounded).
- Recommended `CAUSAL_EVIDENCE_QUERY_MAX_LIMIT = 1000` (aligned with `FunctionalEvidencePersistence` and below DocumentStore `5000`).

Invalid limit → `TypeError` / `ValueError` at validation boundary (mirror `validate_document_query_limit` style).

---

## Cursor contract

| Requirement | Decision |
| ----------- | -------- |
| Opaque to consumers | **YES** — `CausalEvidencePage.next_cursor: str \| None` |
| Query-bound | **YES** — cursor from `(tenant A, provider kafka, task X)` must not work for different tenant, provider, or transport task |
| Provider decodes cursor | **NO** — `CausalTransportScopeProvider` passes `page.next_cursor` unchanged |
| Expose DocumentStore cursor semantics | **NO** — persistence owns envelope; may delegate internally |

Invalid / tampered / query-mismatched cursor → `CausalEvidencePersistenceIntegrityError` (not `NOT_FOUND`). Map from DocumentStore `ValueError` (`document_store_cursor_invalid`, `document_store_cursor_query_mismatch`, `document_store_cursor_authentication_failed`) at the persistence boundary.

Envelope direction (implementation R1): typed causal cursor wrapping authenticated store cursor + query binding (`tenant_id`, `query_kind`, execution or transport scope keys, optional snapshot high-water). Pattern reference: `FunctionalEvidenceQueryCursorCodec`, `ProblemOccurrencePage` store-cursor wrapping.

---

## Ordering decision (frozen)

### Problem

- Canonical order: `(recorded_at, evidence_id)` — `causal_evidence_query_order_key`.
- Current transport index row key: `transport:{provider}:{transport_task_id}:{evidence_id}`.
- Current execution index row key: `exec:{task_id}:{run_id}:{evidence_id}`.
- DocumentStore default paging order = row_key lexicographic ≠ canonical temporal order.
- Page-local sort by `recorded_at` is **forbidden** — timestamps can interleave across pages keyed by `evidence_id`.

### Chosen path: **Option A — index row key includes canonical order**

Encode sortable `recorded_at` into the index row key (same philosophy as `ProblemOccurrence` `_occurrence_row_key` with microsecond sort token), with `evidence_id` tie-break:

```text
transport:{provider}:{transport_task_id}:{recorded_at_sort_token}:{evidence_id}
exec:{task_id}:{run_id}:{recorded_at_sort_token}:{evidence_id}
```

Then DocumentStore prefix query + row_key cursor continuation = globally correct `(recorded_at, evidence_id)` order **without full materialization**.

### Rejected paths

| Option | Verdict |
| ------ | ------- |
| B — DocumentStore `sort=` on data field | Rejected as primary: v1 index payload lacks `recorded_at`; Cassandra adapter does not support cursor queries at all; sort + cursor v2 support is backend-variable |
| C — redefine page order to index order | Rejected — breaks platform-wide canonical causal ordering contract |

### Index projection extension (R1 implementation, not this audit)

**INDEX V2** (`intergrax.causal_evidence.index.v2`):

```text
evidence_id
recorded_at   # ISO-8601 or normalized epoch micros in payload for validation/repair
```

Query uses v2 row-key prefix only for `page_for_*`.

---

## Legacy index compatibility (frozen)

Existing durable rows use **INDEX V1** (`intergrax.causal_evidence.index.v1`): payload `{schema_version, evidence_id}`; row key ends with `evidence_id` only.

| Rule | Decision |
| ---- | -------- |
| Silently skip v1 rows in bounded path | **FORBIDDEN** — loses causal truth |
| Full partition migration in one `page_for_*` call | **FORBIDDEN** |
| v1 decode on unbounded `list_for_*` | **YES** — compatibility facade may still reach v1 via repair or dual-path consume |
| New appends | Create / repair **v2** index on idempotent append (`put_if_absent` + verify) |
| Existing v1-only partitions | Bounded reconciliation job (separate bounded rounds, like problem list index reconciliation); enterprise qualification requires v2 coverage for transport scopes under test |
| v1-only scope + `page_for_*` before repair | Bounded read covers v2 prefix only; if v1 rows exist unrepaired, `candidate_count_exact` must be **false** or persistence returns integrity until reconciliation completes — prefer explicit incomplete snapshot over silent omission |

---

## Snapshot / high-water semantics (frozen)

DocumentStore cursor is continuation-oriented; concurrent appends may appear on later pages unless bounded.

**Minimum guarantee (all backends):**

```text
deterministic monotonic continuation without duplicates for a stable dataset
best-effort append visibility under concurrent writers
```

**Enterprise snapshot (when INDEX V2 row keys align with canonical order):**

On first `page_for_*` call, freeze an upper bound (`row_key_upper_bound` / high-water) from the query start state. Encode in causal cursor envelope. Evidence with canonical order key **after** the high-water is excluded from the query snapshot.

`candidate_count_exact=True` means:

```text
all evidence in the bounded query snapshot / range has been consumed
```

—not merely `next_cursor was None at one moment` under concurrent writes.

Without high-water (or before INDEX V2), document weaker semantics explicitly in conformance tests.

---

## Page integrity semantics

Per page, for each index row:

1. Decode index ref
2. Fetch canonical record
3. Validate evidence_id, tenant, scope (execution or transport)
4. Decode relation record

| Rule | Decision |
| ---- | -------- |
| One corrupt index row on page | **Entire page call fails** (`CausalEvidencePersistenceIntegrityError`) |
| Skip corrupt rows | **FORBIDDEN** |
| Partial page | **FORBIDDEN** |

Same fail-closed semantics as current `_list_indexed`.

---

## In-memory backend (frozen)

Even though test/local only, `InMemoryCausalEvidencePersistence` must implement **true bounded paging**:

- Maintain per-scope index in canonical `(recorded_at, evidence_id)` order on append (incremental insert), **not** sort-all-then-slice per page.
- Opaque query-bound cursor envelope (not raw numeric offset exposed without binding).

---

## Provider budget (frozen defaults)

Separate concepts (do not conflate):

| Concept | Purpose | Frozen default |
| ------- | ------- | -------------- |
| `page_size` | Persistence `limit` per `page_for_transport_task` call | **100** (match `ProblemScopeProvider._OCCURRENCE_PAGE_SIZE`) |
| `max_evidence_examined` | Hard ceiling on total evidence records examined per discovery | **1000** (match `ProblemScopeProvider._MAX_EXAMINED_OCCURRENCES`) |
| `candidate_limit` | Public candidate projection cap (existing request param) | **1..100** unchanged |

**Hard invariant:** `MAX_TRANSPORT_EVIDENCE_EXAMINED` is a finite integer constant. No `0 = unlimited`, no `None = unlimited`, no environment fallback removing the bound.

Provider dedupes `(task_id, run_id)` incrementally per page. Memory scales with bounded examined evidence + distinct candidate scopes.

Provenance for each scope: earliest evidence in canonical order within the bounded query — first evidence seen per scope during ordered page walk is canonical earliest.

---

## Truncated scan classification (frozen)

Mirror hardened `ProblemScopeProvider` semantics:

| Condition | Status | `candidate_count_exact` |
| --------- | ------ | ----------------------- |
| Full scan complete | Per distinct scope count (0 → `NOT_FOUND` if no evidence; 1 → `RESOLVED`; ≥2 → `AMBIGUOUS`) | **True** |
| Truncated (`max_evidence_examined` hit while `next_cursor != None`) and 0–1 known scopes | `INSUFFICIENT_EVIDENCE` | **False** |
| Truncated and ≥2 distinct execution scopes already found | `AMBIGUOUS` | **False** (`candidate_count` = known lower bound) |
| Full scan, 5 distinct scopes | `AMBIGUOUS`, `candidate_count=5` | **True** |

Current `CausalTransportScopeProvider` always sets `candidate_count_exact=True` for ambiguous multi-scope results; this must change when bounded paging lands.

---

## Diagnostics rule (DG-002 closure)

After implementation, `CausalTransportScopeProvider` **MUST NOT** call `list_for_transport_task()`. It must call `page_for_transport_task()` incrementally until budget exhausted or `next_cursor is None`.

`ExecutionReconstructor` **remains** on `list_for_execution()` during DG-002. `page_for_execution` exists for abstraction symmetry and future DG-004 / scale work.

---

## Architectural constraints (confirmed)

| Constraint | Status |
| ---------- | ------ |
| Semantic owner = `CausalEvidencePersistence` | **YES** |
| No new diagnostics-specific persistence spine | **YES** |
| No queue coupling in page API | **YES** |
| Full-store materialization on bounded path | **FORBIDDEN** |
| Strong typing (no public `Any` / dict pages) | **Required** |

---

## DocumentStore primitive reuse

Reuse existing `DocumentStore.query(partition_key, *, limit, row_key_prefix, cursor, row_key_upper_bound, sort, data_equalities) → DocumentQueryPageV1`. Do not invent backend-specific pagination.

**Backend note:** Cassandra adapter rejects `cursor is not None` today. INDEX V2 row-key ordering minimizes dependence on sort+cursor v2 and is compatible with prefix+limit backends; multi-page causal reads on Cassandra remain a pre-existing platform limitation until that adapter gains cursor support.

---

## Recommended implementation sequence

| Slice | Scope |
| ----- | ----- |
| **R1** | `CausalEvidencePage` + `page_for_execution` / `page_for_transport_task` + INDEX V2 + backends + conformance + list facades |
| **R2** | `CausalTransportScopeProvider` bounded incremental consumption + truncation semantics + static guard against `list_for_transport_task` in diagnostics |
| **R3** | DG-002 transport bounded-read qualification |
| **R4** | DG-002 final closure record |

**Recommendation:** keep **R1** and **R2** separate — INDEX V2 + cursor envelope + dual-backend conformance is non-trivial; provider migration should land on a frozen paging contract.

---

## Stop conditions evaluated

| Condition | Result |
| --------- | ------ |
| DocumentStore cursor cannot preserve required global ordering | **Not a blocker** — Option A (canonical row key) avoids sort-dependent cursors |
| Canonical order cannot be retained without index migration | **Migration required** — INDEX V2 + reconciliation; not a stop |
| Cursor semantics differ materially across DocumentStore backends | **Manageable** — opaque causal envelope + row-key ordering; document Cassandra cursor gap |
| Paging requires queue-specific knowledge | **NO** |
| Bounded API cannot be added compatibly | **NO** — add page API + retain list facades |

---

## Production files changed

**NONE** (audit / design only).

---

## Final statement

```text
DG-002 TRANSPORT BOUNDED-READ GAP
= CONFIRMED

SEMANTIC OWNER
= CausalEvidencePersistence

PAGED READ CONTRACT
= REQUIRED

EXISTING LIST CONTRACT
= RETAINED FOR COMPATIBILITY

DOCUMENTSTORE CURSOR PRIMITIVE
= REUSED

CANONICAL ORDER
= MUST BE PRESERVED WITHOUT FULL MATERIALIZATION

TRANSPORT PROVIDER
= MUST MIGRATE TO INCREMENTAL PAGED READ

EXECUTION RECONSTRUCTOR
= OUT OF DG-002 MIGRATION SCOPE

CANDIDATE EXACTNESS
= FAIL-CLOSED UNDER TRUNCATION

FULL-STORE MATERIALIZATION
= FORBIDDEN ON BOUNDED PATH

DG-002 ENTERPRISE BLOCKER
= DESIGN READY FOR IMPLEMENTATION

NEXT
= DG-002 CAUSAL EVIDENCE PAGING CONTRACT IMPLEMENTATION (R1)
```
