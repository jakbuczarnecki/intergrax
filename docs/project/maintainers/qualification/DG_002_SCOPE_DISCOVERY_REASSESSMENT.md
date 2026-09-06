# DG-002 Scope Discovery — Qualification Gap Reassessment (R1)

**Verdict:** PASS (reassessment frozen; enterprise closure **not** claimed)

**Date:** 2026-09-06

**Branch:** `development`

**Start HEAD:** `8dd5c949cbbc9dce036225903a269f36918a7173`

**Task:** `DG-002-QUALIFICATION-GAP-REASSESSMENT-R1` — review / qualification decision only; no production changes.

---

## DG-002 core definition (frozen)

**DG-002 CORE** = resolve a canonical operator reference into an existing diagnostic execution scope (or explicit non-resolution status) without identity minting.

Supported canonical reference kinds (frozen):

| Kind | Reference | Provider | Persistence path |
| ---- | --------- | -------- | ---------------- |
| `PROBLEM` | `ProblemId` | `ProblemScopeProvider` | `ProblemPersistence` + `ProblemOccurrencePersistence` |
| `TRANSPORT` | `{provider, transport_task_id}` | `CausalTransportScopeProvider` | `CausalEvidencePersistence.list_for_transport_task` |
| `EVENT` | `EventId` | `RuntimeEventScopeProvider` | `RuntimeEventPersistence.get_by_event_id` |

Each provider resolves to `DiagnosticExecutionScope` (`ExecutionDiagnosticSubjectRef`) or an explicit status (`NOT_FOUND`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, `NON_EXECUTION_SUBJECT`, `PROVIDER_UNAVAILABLE`).

**Not in minimal DG-002 core:** `correlation_id`, `ExecutionId`, raw `TaskId`/`RunId` discovery.

---

## Enterprise closure definition (frozen)

Diagnostic Scope Discovery is **enterprise-qualified** when:

1. Discovery core is provider-neutral.
2. Supported reference kinds are typed and explicit.
3. All supported providers are tenant-safe.
4. Ambiguity is explicit (no first-match-wins).
5. Identity is never minted.
6. **Reads are bounded / exact** at the public persistence boundary.
7. `unavailable` ≠ `NOT_FOUND`.
8. Persistence corruption fails closed.
9. Provider addition does not require core changes.
10. Supported production paths have qualification evidence.

---

## Provider qualification matrix

| Property | Problem | Transport | Event |
| -------- | ------: | --------: | ----: |
| typed ref | PASS | PASS | PASS |
| tenant-safe | PASS | PASS | PASS |
| provider-neutral | PASS | PASS | PASS |
| no minting | PASS | PASS | PASS |
| ambiguity honest | PASS | PASS | N/A (exact 0/1) |
| deterministic | PASS | PASS | PASS |
| **bounded read** | PASS | **FAIL / OPEN** | PASS |
| backend-independent | PASS | PASS | PASS |
| failure normalization | PASS | PASS | PASS |

---

## Problem provider — enterprise semantics: PASS

`ProblemScopeProvider` (`intergrax/runtime/diagnostics/providers/problem_scope_provider.py`):

- Tenant-scoped `ProblemPersistence.get`.
- Paginated `ProblemOccurrencePersistence.query_occurrences` with `_OCCURRENCE_PAGE_SIZE = 100` and `_MAX_EXAMINED_OCCURRENCES = 1000`.
- Explicit `AMBIGUOUS` when multiple execution scopes; `candidate_count` truth preserved.
- Truncation sets `candidate_count_exact=False` and `INSUFFICIENT_EVIDENCE` when classification cannot complete within examination bound.
- No first-match-wins; no identity minting.

---

## Event provider — enterprise semantics: PASS

`RuntimeEventScopeProvider` (`intergrax/runtime/diagnostics/providers/runtime_event_scope_provider.py`):

- Exact `RuntimeEventPersistence.get_by_event_id(tenant_id, event_id)` → 0 or 1 `PositionedRuntimeEvent`.
- Tenant isolation enforced; integrity errors fail closed.
- Indexed lookup backends qualified separately (`test_event_id_persistence_semantics`, `test_document_backed_event_id_index`, crash-recovery proofs).
- No provider backend coupling in discovery layer.

---

## Transport provider — functional semantics: PASS; bounded read: FAIL

`CausalTransportScopeProvider` functional semantics are qualified:

- Tenant-scoped causal evidence validation.
- Accepts only `CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION`.
- Deterministic dedupe by `(task_id, run_id)`; explicit `AMBIGUOUS` for 0/1/N execution scopes.
- No queue implementation coupling; no identity minting.

**Bounded read: FAIL.** Public persistence contract:

```python
def list_for_transport_task(
    self, *, tenant_id: str, provider: str, transport_task_id: str,
) -> tuple[PlatformCausalEvidence, ...]:
```

No `limit`, no `cursor`, no page type. Caller cannot bound materialized evidence.

**Implementations audited (all):**

| Implementation | Bounded public read? |
| -------------- | ------------------- |
| `InMemoryCausalEvidencePersistence` | NO — returns all matching evidence IDs |
| `DocumentStoreCausalEvidencePersistence` | NO — `_list_indexed` paginates internally (`_QUERY_PAGE_LIMIT = 5000`) but accumulates **all** index rows into `documents` before decode and return |

`CausalTransportScopeProvider._list_transport_evidence` calls the unbounded contract and materializes the full tuple before `candidate_limit` projection.

### `candidate_limit` does NOT bound storage read

`candidate_limit` (default 10, max 100) only bounds the **public candidate projection** in `_classify_execution_scopes` (`ordered_candidates[:candidate_limit]`) and in `DiagnosticScopeDiscoveryService._project_provider_result`. It does **not** bound `CausalEvidencePersistence.list_for_transport_task()`.

Rejected fake boundedness:

- `tuple(persistence.list_for_transport_task(...))[:100]` after full materialization.
- Provider early-stop after N candidates when persistence already loaded all evidence.

**Internal pagination ≠ bounded public read.**

---

## Correlation — deferred extension (not core blocker)

### Is `correlation_id` a canonical platform unique identity?

**NO.** Platform-wide audit shows heterogeneous usage:

- Task/run correlation (`emit_context.effective_correlation_id`, trace bridge).
- Session / trace correlation (`context_engine`, OTLP).
- Transport metadata (`problem_signal`, observability export).
- Application-specific correlation (qualification proofs, LKW certification scripts).

No single frozen semantic owner or cardinality contract.

### Hard invariant `correlation_id → exactly one TaskId/RunId`?

**NO.** Default model: `correlation_id → 0..N` executions/events unless a specific subsystem proves uniqueness (none frozen platform-wide).

### Canonical persistence lookup `list_by_correlation_id(...)`?

**NOT PRESENT.** Repo-wide search: zero matches for `list_by_correlation_id` / `get_by_correlation_id`.

### Required for minimal DG-002 closure?

**NO.** Frozen decision:

```text
CORRELATION DISCOVERY = DEFERRED DG-002 EXTENSION
```

Pending explicit canonical correlation identity / cardinality / persistence contract and separate qualification.

---

## ExecutionId — not required for minimal DG-002 closure

`RuntimeEvent` carries `execution_id`, but operator scope discovery has:

- No `ExecutionId` reference kind in `DiagnosticScopeReferenceKind`.
- No canonical `get_by_execution_id` discovery provider.
- No demonstrated operator need for ExecutionId-as-reference in DG-002 scope.

Frozen: **DEFERRED / NOT CORE CLOSURE REQUIREMENT.**

---

## Frozen reference enum

`DiagnosticScopeReferenceKind` remains:

```text
PROBLEM | TRANSPORT | EVENT
```

No `CORRELATION`, `EXECUTION`, `TASK`, or `RUN` reference kinds without separate design + qualification.

---

## Relationship to other gaps

| Gap | Relationship |
| --- | ------------ |
| **DG-004** | Separate. DG-002: transport ref → execution scope(s). DG-004: reconstructed diagnosis proves transport → execution causal continuity. Transport discovery does **not** close DG-004. |
| **DG-005** | Separate. EventId lookup ≠ split-host topology qualification. DG-005 remains qualification candidate. |

---

## Current DG-002 status (frozen)

| Layer | Status |
| ----- | ------ |
| DG-002 core architecture | **QUALIFIED** |
| DG-002 functional semantics (Problem / Transport / Event) | **QUALIFIED** |
| DG-002 enterprise qualification | **NOT YET CLOSED** |
| Sole remaining core blocker | **TRANSPORT BOUNDED READ** |

Status in ledger: **PARTIALLY ADDRESSED** (unchanged until transport boundedness is fixed and qualified).

---

## Next task

```text
DG-002-TRANSPORT-BOUNDED-READ-HARDENING-AUDIT-R1
```

Audit/design only (not implementation):

- All `CausalEvidencePersistence` implementations and callers.
- Deterministic page order, cursor semantics, compatibility, integrity semantics.
- Potential future shape (not frozen): paginated `list_for_transport_task` returning a page type with `limit` + `cursor`.

---

## Production files changed

**NONE** (documentation-only reassessment).

## H1 canonical rerun

**NO**
