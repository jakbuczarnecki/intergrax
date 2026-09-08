# DG-002 Diagnostic Scope Discovery — Final Closure (R4)

**Verdict:** PASS

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `4228769fbb08be2465763dd612a8a8b540f4cef3`

**Review HEAD (pre-docs):** `4228769fbb08be2465763dd612a8a8b540f4cef3`

**Task:** `DG-002-FINAL-CLOSURE-REVIEW-R4` — review-only final closure; no production or test changes.

---

## 1. Verdict

```text
DG-002 DIAGNOSTIC SCOPE DISCOVERY = CLOSED / QUALIFIED
```

Core Diagnostic Scope Discovery is enterprise-qualified for **ProblemId**, **transport task reference**, and **EventId**. Correlation discovery and ExecutionId discovery remain deferred extensions and are not closure blockers.

---

## 2. Scope

Frozen DG-002 core scope:

```text
Operator/API
  ↓
DiagnosticScopeDiscoveryService
  ├── ProblemScopeProvider
  ├── CausalTransportScopeProvider
  └── RuntimeEventScopeProvider
  ↓
DiagnosticScopeDiscoveryResult
  ↓
DiagnosticExecutionScope | DiagnosticSignalSubjectScope
  ↓
existing DiagnosticOrchestrator
```

**Supported canonical core references:** `PROBLEM`, `TRANSPORT`, `EVENT`.

**Explicit non-core / deferred:** correlation_id discovery, ExecutionId discovery, arbitrary unstructured correlation.

---

## 3. Frozen architecture

| Invariant | Status |
| --------- | ------ |
| Discovery = read-only / derived | PASS |
| Identity minting = none | PASS |
| Tenant-scoped lookup = mandatory | PASS |
| First-match-wins = forbidden | PASS |
| Ambiguous = explicit result | PASS |
| Provider unavailable ≠ NOT_FOUND | PASS |
| Core service = provider-neutral | PASS |
| No new diagnostic mapping truth | PASS |

---

## 4. Supported references

| Kind | Reference | Provider | Persistence path |
| ---- | --------- | -------- | ---------------- |
| `PROBLEM` | `ProblemId` | `ProblemScopeProvider` | `ProblemPersistence` + `ProblemOccurrencePersistence` |
| `TRANSPORT` | `{provider, transport_task_id}` | `CausalTransportScopeProvider` | `CausalEvidencePersistence.page_for_transport_task` |
| `EVENT` | `EventId` | `RuntimeEventScopeProvider` | `RuntimeEventPersistence.get_by_event_id` |

---

## 5. Qualification chain

All canonical implementation SHAs are ancestors of review HEAD (`4228769f`):

| Milestone | SHA |
| --------- | --- |
| ProblemId scope discovery (Slice 1) | `f2198be56` |
| Problem provider remediation | `01dcc7783222089998773bea4d4937ad67638632` |
| EventId persistence hardening | `d33fb6f8a6864a8096e3b8a36e5b0eb18d43cd0a` |
| RuntimeEventScopeProvider | `d3208983b649bf4311a4245f078b2c9458eab774` |
| DG-002 reassessment (R1) | `bc49f67be07677e95324f3d7ed2839569e74aad0` |
| Causal paging R1 | `d0ed4a0b3b7ec975ddeb5392f965250b0b867a6d` |
| Transport provider R2 remediation | `3b4fc73afdc200957be23c6b5042b236efef6bbc` |
| Transport bounded read R3 qualification | `0a6e1af1cb0c33d2455406f548dd4dc00bbcdbbd` |

Prior qualification docs: [`DG_002_SCOPE_DISCOVERY_REASSESSMENT.md`](DG_002_SCOPE_DISCOVERY_REASSESSMENT.md), [`DG_002_TRANSPORT_BOUNDED_READ_HARDENING_AUDIT.md`](DG_002_TRANSPORT_BOUNDED_READ_HARDENING_AUDIT.md), [`DG_002_TRANSPORT_BOUNDED_READ_R3_QUALIFICATION.md`](DG_002_TRANSPORT_BOUNDED_READ_R3_QUALIFICATION.md).

---

## 6. Provider-by-provider result

### Problem scope discovery — QUALIFIED

- Tenant-scoped `ProblemPersistence.get`.
- Paginated `ProblemOccurrencePersistence.query_occurrences`; page size 100; hard max 1000 examined.
- Explicit `AMBIGUOUS` for multiple execution scopes; truncation → `INSUFFICIENT_EVIDENCE` with `candidate_count_exact=False`.
- No first-match-wins; no identity minting; provider-specific persistence errors translated at provider boundary.

### Transport scope discovery — ENTERPRISE QUALIFIED

- `page_for_transport_task` only; no `list_for_transport_task` in production provider.
- Page size 100; hard max 1000 evidence examined.
- Exactness matrix per R3 qualification preserved.
- Cursor cycle protection; oversized page fail-closed; earliest provenance preserved.
- No queue/backend coupling.

### EventId scope discovery — QUALIFIED

- Exact `RuntimeEventPersistence.get_by_event_id(tenant_id, event_id)` → 0 or 1.
- Tenant isolation enforced; missing → `NOT_FOUND`; existing → `RESOLVED` exact 1.
- EventId ownership remains semantic responsibility of `RuntimeEventPersistence`.
- No full-store scans; no duplicate event index in diagnostics layer.

---

## 7. Core invariants

`DiagnosticScopeDiscoveryService` dispatches through `DiagnosticScopeDiscoveryProvider` contract only. Core has no imports of concrete Problem, CausalEvidence, or RuntimeEvent persistence implementations. Three providers coexist without core change. Unexpected provider errors are not hidden (`DiagnosticScopeProviderUnavailableError` → `PROVIDER_UNAVAILABLE`; integrity errors propagate fail-closed).

---

## 8. Enterprise boundedness

| Path | Boundedness |
| ---- | ----------- |
| Problem | Paginated occurrence history; max 1000 examined |
| Transport | Paged causal evidence scan; hard max 1000 examined |
| Event | Exact EventId lookup (no scan) |

---

## 9. Tenant isolation

All three providers require mandatory `tenant_id`. No global lookup, cross-tenant scan, or fallback without tenant. Evidence integrity checks enforce tenant match on all persisted records consumed.

---

## 10. Identity authority

DG-002 production discovery paths do not mint `TaskId`, `RunId`, `AttemptId`, or `ExecutionId`. Providers reconstruct execution scope references from persisted authority only. **Execution runtime remains identity authority.**

---

## 11. Pluginability

```text
THREE-PROVIDER PLUGINABILITY = QUALIFIED
```

`DiagnosticScopeDiscoveryProviderRegistry` resolves provider by reference kind. Adding a fourth provider requires no core service change.

---

## 12. Deferred extensions

| Extension | Status |
| --------- | ------ |
| Correlation_id discovery | DEFERRED EXTENSION / NOT CORE |
| ExecutionId discovery | DEFERRED / NOT CORE |

`correlation_id` is heterogeneous with 0..N cardinality and is not canonical unique execution identity. Deferred correlation discovery is not a defect in closed core.

---

## 13. Non-claims

- **DG-002 CLOSED does NOT mean** arbitrary `correlation_id` discovery exists.
- **DG-002 CLOSED does NOT mean** ExecutionId discovery is supported.
- **DG-002 CLOSED does NOT mean** transport causal reconstruction continuity (DG-004) is solved.
- **DG-002 CLOSED does NOT mean** runtime event persistence topology is globally qualified (DG-005).

---

## 14. Relationship to DG-004 / DG-005

| Gap | Relationship |
| --- | ------------ |
| **DG-004** (async transport causal continuity in execution diagnosis) | **UNCHANGED / OPEN** — does not block DG-002 closure. DG-002 resolves transport ref → execution scope(s); DG-004 concerns reconstruction continuity proof. No coupling discovered. |
| **DG-005** (runtime event persistence topology qualification) | **UNCHANGED** — does not block DG-002 closure. EventId discovery uses canonical `RuntimeEventPersistence.get_by_event_id` capability as designed; no unproven topology invariant blocks discovery. |

---

## 15. Backwards compatibility

- Explicit `DiagnosticExecutionScope(TaskId, RunId)` fast path in `DiagnosticOrchestrator` remains unchanged.
- `DiagnosticSignalSubjectScope` and non-execution subject semantics preserved.
- `PlatformProblemSignal` without TaskId/RunId does not cause identity fabrication.

---

## 16. Static architecture audit

Production DG-002 code review confirms:

- No concrete backend imports in providers/core (structural test `test_discovery_core_has_no_forbidden_imports`).
- No `list_for_transport_task` in transport provider production path.
- No `DiagnosticScopeMappingStore`, `DiagnosticReferenceIndexStore`, or `DiagnosticExecutionMappingStore`.
- No reflection or dynamic import hacks in DG-002 abstractions.
- No first-match-wins behavior.

---

## 17. Exact regression commands

```bash
uv sync --extra dev-ci --frozen

# Focused three-provider tests
uv run pytest \
  tests/unit/runtime/diagnostics/test_problem_scope_provider.py \
  tests/unit/runtime/diagnostics/test_causal_transport_scope_provider.py \
  tests/unit/runtime/diagnostics/test_runtime_event_scope_provider.py \
  tests/unit/runtime/diagnostics/test_diagnostic_scope_discovery_registry.py \
  tests/unit/runtime/diagnostics/test_diagnostic_scope_discovery_architecture.py \
  -q --basetemp=.tmp/session/dg002-r4/pytest-basetemp

# Persistence regression
uv run pytest \
  tests/unit/runtime/observability/ \
  tests/unit/runtime/architecture/test_causal_evidence_paging_architecture.py \
  -q --basetemp=.tmp/session/dg002-r4/pytest-basetemp-obs

# EventId persistence
uv run pytest \
  tests/unit/runtime/events/test_runtime_event_persistence.py \
  tests/unit/runtime/events/test_event_id_persistence_semantics.py \
  tests/unit/runtime/events/test_document_backed_event_id_index.py \
  tests/unit/runtime/events/test_event_id_ownership_crash_recovery.py \
  -q --basetemp=.tmp/session/dg002-r4/pytest-basetemp

# Problem persistence
uv run pytest \
  tests/unit/runtime/diagnostics/test_problem_scope_provider.py \
  tests/unit/runtime/diagnostics/test_problem_persistence_conformance.py \
  tests/unit/runtime/diagnostics/test_diag_enterprise_1_scalable_problem_reads.py \
  tests/unit/runtime/diagnostics/test_durable_problem_persistence.py \
  -q --basetemp=.tmp/session/dg002-r4/pytest-basetemp

# Background admission
uv run pytest \
  tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py \
  -q --basetemp=.tmp/session/dg002-r4/pytest-basetemp-bg

# H1 architecture gates (not canonical H1-R3 rerun)
uv run pytest \
  tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py \
  -q --basetemp=.tmp/session/dg002-r4/pytest-basetemp-h1
```

---

## 18. Exact test counts

| Suite | Result |
| ----- | ------ |
| Focused three-provider regression | **105 passed**, 0 failed, 0 skipped |
| Persistence regression (observability + causal paging) | **306 passed**, 0 failed, 0 skipped |
| EventId persistence (4 files) | **76 passed**, 0 failed, 0 skipped |
| Problem persistence (4 files) | **64 passed**, 0 failed, 0 skipped |
| Background causal admission | **10 passed**, 0 failed, 0 skipped |
| H1 architecture gates | **3 passed**, 0 failed, 0 skipped |
| Full closure combined regression | **539 passed**, 0 failed, 0 skipped |

`--ignore` used: **NO**. New skips: **NONE**. H1 canonical R3 rerun: **NO**.

---

## 19. Final closure statement

```text
DG-002 FINAL CLOSURE REVIEW R4 = PASS ✅

DG-002 DIAGNOSTIC SCOPE DISCOVERY = CLOSED / QUALIFIED ✅

SUPPORTED CORE REFERENCES = PROBLEM | TRANSPORT | EVENT ✅

PROBLEM SCOPE DISCOVERY = QUALIFIED ✅
TRANSPORT SCOPE DISCOVERY = ENTERPRISE QUALIFIED ✅
EVENT-ID SCOPE DISCOVERY = QUALIFIED ✅
THREE-PROVIDER PLUGINABILITY = QUALIFIED ✅
CORE DISCOVERY SERVICE = PROVIDER-NEUTRAL ✅
TENANT ISOLATION = QUALIFIED ✅
IDENTITY MINTING = NONE ✅
FIRST-MATCH-WINS = FORBIDDEN ✅
AMBIGUITY = EXPLICIT ✅
CANDIDATE EXACTNESS = HONEST ✅
TRANSPORT READ = HARD-BOUNDED 1000 ✅
FULL TRANSPORT HISTORY MATERIALIZATION = ELIMINATED ✅
NEW DIAGNOSTIC MAPPING TRUTH = NONE ✅

CORRELATION DISCOVERY = DEFERRED EXTENSION / NOT CORE
EXECUTION-ID DISCOVERY = DEFERRED / NOT CORE

DG-004 = UNCHANGED / OPEN
DG-005 = UNCHANGED

REMAINING DG-002 CORE BLOCKERS = NONE

NEXT = NEXT OPEN DIAGNOSTIC GAP FROM LEDGER
```
