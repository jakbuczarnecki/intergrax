# DG-002 Transport Bounded Read — Qualification (R3)

**Verdict:** PASS

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `c9c8cca85af68f8729791c8eb313d5d3cb193c91`

**Qualification HEAD (pre-docs):** `c9c8cca85af68f8729791c8eb313d5d3cb193c91`

**Qualified implementation SHA — persistence R1:** `d0ed4a0b3b7ec975ddeb5392f965250b0b867a6d`

**Qualified implementation SHA — provider R2 remediation:** `3b4fc73afdc200957be23c6b5042b236efef6bbc`

**Both ancestors of qualification HEAD:** YES

---

## Scope

Qualification subject: **DG-002 TRANSPORT BOUNDED READ** only.

```text
TransportScopeReference
→ CausalTransportScopeProvider
→ CausalEvidencePersistence.page_for_transport_task()
→ bounded ordered causal evidence pages
→ incremental execution-scope deduction
→ DiagnosticScopeDiscoveryResult
```

**Not qualified:** DG-004 causal reconstruction continuity, DG-005 topology, queue runtime, worker delivery, ExecutionReconstructor paging.

---

## Frozen invariants — qualification evidence

| ID | Invariant | Evidence |
| -- | --------- | -------- |
| Q1 | Provider uses `page_for_transport_task` only | `test_provider_source_uses_page_api_only`, `test_page_api_only_structural_guard`; production source line 121 |
| Q2 | Provider never uses `list_for_transport_task` | Structural grep: absent in `causal_transport_scope_provider.py`; `test_page_api_only_structural_guard` asserts `list_calls == 0` |
| Q3 | Persistence page is bounded | `test_causal_evidence_paging_architecture.py`; observability persistence suite (306 passed) |
| Q4 | Provider request page bound ≤ remaining budget | `page_limit = min(100, remaining)` at provider line 146; `test_remaining_budget_uses_reduced_page_limit` |
| Q5 | Provider validates returned page ≤ requested page | Lines 155–158; `test_oversized_page_raises_integrity_error` |
| Q6 | Max evidence examined ≤ 1000 | `_MAX_EXAMINED_CAUSAL_EVIDENCE = 1000`; `test_exactly_thousand_evidence_complete_scan`, `test_read_count_bound_with_large_available_history` |
| Q7 | Malformed oversized page fails before processing | `examined_count` increments only inside `for evidence in page.items` after size check; oversized tests assert `call_count == 1` or `page_calls == 11` (no partial consume) |
| Q8 | `candidate_limit` does not affect evidence scan | `test_candidate_limit_preserves_truth_count`; `test_hundred_one_evidence_same_scope_uses_two_pages` independent of limit |
| Q9 | Canonical order preserved | `_OffsetPagedTransportPersistence` offset paging; multi-page ambiguity tests |
| Q10 | Earliest provenance preserved | `test_earliest_provenance_survives_duplicate_on_later_page` |
| Q11 | Duplicate scopes dedupe incrementally | `test_duplicate_scope_across_pages_dedupes` |
| Q12 | Full scan exactness true | `test_exactly_thousand_evidence_complete_scan` → `candidate_count_exact is True` |
| Q13 | Truncated 0/1 → INSUFFICIENT_EVIDENCE exact=false | `test_truncated_one_scope_is_insufficient_inexact` |
| Q14 | Truncated ≥2 → AMBIGUOUS exact=false | `test_truncated_two_scopes_is_ambiguous_inexact` |
| Q15 | Tenant isolation hard | `test_tenant_mismatch_raises_integrity_error`; provider `_validate_evidence_integrity` lines 251–262 (evidence/source/target tenant) |
| Q16 | Persistence integrity fail-closed | `test_persistence_integrity_maps_to_provider_integrity`, `test_integrity_on_second_page_fails_whole_discover` |
| Q17 | Provider unavailable distinct from NOT_FOUND | `test_availability_on_second_page_fails_whole_discover`, `test_os_error_maps_to_provider_unavailable` |
| Q18 | No cursor decoding in provider | Provider passes opaque `cursor` to persistence; no decode/import of cursor helpers |
| Q19 | Cursor cycle protection | `test_repeated_cursor_raises_integrity_error`; provider lines 168–176 |
| Q20 | No backend/queue coupling | Structural audit: no `DocumentStore`, `Kafka`, `Celery`, `RabbitMQ` in production provider |

---

## Exactness matrix

| Scan | Distinct scopes | Status | exact |
| ---- | --------------: | ------ | ----: |
| complete | 0 | NOT_FOUND | true |
| complete | 1 | RESOLVED | true |
| complete | ≥2 | AMBIGUOUS | true |
| truncated | 0 | INSUFFICIENT_EVIDENCE | false |
| truncated | 1 | INSUFFICIENT_EVIDENCE | false |
| truncated | ≥2 | AMBIGUOUS | false |

Confirmed by `_classify_execution_scopes` and transport provider test suite.

---

## Hard 1000 bound proof

Production constants:

```text
_CAUSAL_EVIDENCE_PAGE_SIZE = 100
_MAX_EXAMINED_CAUSAL_EVIDENCE = 1000
```

Per iteration: `page_limit = min(100, remaining)` where `remaining = 1000 - examined_count`.

Validation before consume: `if len(page.items) > page_limit: raise integrity`.

Therefore: `examined_count ≤ 1000` for every provider execution.

Test evidence: `test_budget_caps_page_calls_at_ten` (12000 records → 10 page calls), `test_read_count_bound_with_large_available_history` (15000 records → requested_limits == 1000).

---

## Oversized page proof

| Case | Test | Result |
| ---- | ---- | ------ |
| requested=100 returned=101 | `test_oversized_page_raises_integrity_error` | integrity, no partial consume |
| remaining=50 returned=51 | `test_oversized_remaining_budget_page_raises_integrity_error` | integrity at page 11 |
| remaining=50 returned=50 | `test_exact_remaining_budget_page_completes_scan` | RESOLVED, exact=true |

---

## Large-history bounded proof

Conceptual >10,000 matching evidence:

- `test_budget_caps_page_calls_at_ten` — 12000 records, max 10 page calls
- `test_read_count_bound_with_large_available_history` — 15000 records, sum(requested limits) == 1000
- `test_truncated_one_scope_is_insufficient_inexact` — 1000 + trailing cursor → truncated INSUFFICIENT_EVIDENCE

No full history materialization.

---

## Real paged persistence integration

| Backend | Test | Result |
| ------- | ---- | ------ |
| `InMemoryCausalEvidencePersistence` | `test_in_memory_paged_integration_crosses_page_boundary` (101 records) | PASS |
| `DocumentStoreCausalEvidencePersistence` | `test_document_store_paged_integration_crosses_page_boundary` (101 records) | PASS |

---

## Structural source audit

**Production file:** `intergrax/runtime/diagnostics/providers/causal_transport_scope_provider.py`

**Present:** `page_for_transport_task(`

**Absent (production):** `list_for_transport_task(`, `DocumentStore`, `InMemoryDocumentStore`, `causal_evidence_index`, `causal_evidence_query_cursor`, `Kafka`, `Celery`, `RabbitMQ`

**Persistence ownership:** Provider imports only `CausalEvidencePersistence` protocol — no concrete backend imports.

**Unexpected exceptions:** No blanket `except Exception`; only `CausalEvidencePersistenceIntegrityError`, `ConnectionError`, `TimeoutError`, `OSError`.

---

## Commands and counts

Environment prep:

```bash
uv sync --extra dev-ci --frozen
```

### §7 Transport provider

```bash
uv run pytest tests/unit/runtime/diagnostics/test_causal_transport_scope_provider.py -q
```

**Result:** 44 passed, 0 failed, 0 skipped

### §8 Persistence support

```bash
uv run pytest tests/unit/runtime/observability/ tests/unit/runtime/architecture/test_causal_evidence_paging_architecture.py -q
```

**Canonical run:** 272 passed, 34 ERRORS (Windows `build/pytest-basetemp` PermissionError — environmental, not assertion failures)

**Confirmatory rerun:**

```bash
uv run pytest tests/unit/runtime/observability/ tests/unit/runtime/architecture/test_causal_evidence_paging_architecture.py -q --basetemp=.tmp/session/dg002-r3/pytest-basetemp
```

**Result:** 306 passed, 0 failed, 0 skipped

### §9 Background admission

```bash
uv run pytest tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py -q
```

**Result:** 10 passed, 0 failed, 0 skipped

### §10 Three-provider regression

```bash
uv run pytest tests/unit/runtime/diagnostics/test_problem_scope_provider.py tests/unit/runtime/diagnostics/test_runtime_event_scope_provider.py tests/unit/runtime/diagnostics/test_diagnostic_scope_discovery_registry.py tests/unit/runtime/diagnostics/test_diagnostic_scope_discovery_architecture.py -q
```

**Result:** 61 passed, 0 failed, 0 skipped

### §11 H1 architecture gates

```bash
uv run pytest tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py -q
```

**Result:** 3 passed, 0 failed, 0 skipped

**H1 canonical rerun:** NO

### §12 Full R3 qualification regression

```bash
uv run pytest tests/unit/runtime/diagnostics/test_causal_transport_scope_provider.py tests/unit/runtime/diagnostics/test_diagnostic_scope_discovery_registry.py tests/unit/runtime/diagnostics/test_diagnostic_scope_discovery_architecture.py tests/unit/runtime/diagnostics/test_problem_scope_provider.py tests/unit/runtime/diagnostics/test_runtime_event_scope_provider.py tests/unit/runtime/observability/ tests/unit/runtime/background_execution/ tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py tests/unit/runtime/architecture/test_causal_evidence_paging_architecture.py -q --basetemp=.tmp/session/dg002-r3/pytest-basetemp
```

**Result:** 506 passed, 0 failed, 0 skipped

**`--ignore` used:** NO

**New skips:** NONE

---

## Production / test changes during R3

```bash
git diff c9c8cca85af68f8729791c8eb313d5d3cb193c91 -- intergrax/runtime/
```

**Result:** EMPTY

Unrelated parallel work preserved (not staged): `applications/local_workspace_application/workspaces/connected_source_discovery.py`, `intergrax/marketplace/service.py`

---

## Non-claims

- Does **not** qualify DG-004 (transport causal continuity in ExecutionReconstructor)
- Does **not** qualify DG-005 (runtime event persistence topology)
- Does **not** close full DG-002 (Problem/Event provider final closure review pending)
- Does **not** qualify queue runtime or worker delivery paths

---

## Relation to DG-004 / DG-005

| Gap | Status |
| --- | ------ |
| DG-004 | UNCHANGED / OPEN |
| DG-005 | UNCHANGED |

---

## Qualification statement

```text
DG-002 TRANSPORT BOUNDED READ R3 = QUALIFIED ✅
CAUSAL TRANSPORT SCOPE PROVIDER = ENTERPRISE BOUNDED ✅
PAGED PERSISTENCE = QUALIFIED ✅
MAX EVIDENCE EXAMINED = HARD 1000 ✅
FULL HISTORY MATERIALIZATION = ELIMINATED ✅
CANONICAL ORDER = PRESERVED ✅
INCREMENTAL DEDUPE = QUALIFIED ✅
TRUNCATION = FAIL-CLOSED ✅
CANDIDATE EXACTNESS = HONEST ✅
TENANT ISOLATION = QUALIFIED ✅
THREE-PROVIDER REGRESSION = PASS ✅
DG-002 ENTERPRISE BLOCKER = REMOVED ✅
DG-002 = READY FOR FINAL CLOSURE REVIEW
NEXT = DG-002 FINAL CLOSURE REVIEW
```
