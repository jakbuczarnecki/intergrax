# INTEGRAx-NPSC-5F-V1-V2-PERSISTENCE-CONFLICT-HARDENING

**Task:** INTEGRAx-NPSC-5F-V1-V2-PERSISTENCE-CONFLICT-HARDENING  
**Compatibility baseline:** `91f6c864087756b2995fc13a53f383637c8f74e9`  
**Scoped reopen:** `fe2b1510ddd87ebd809de14f9ed9d0a459bfb5b8`

## Metadata

| Field | Value |
|-------|-------|
| Domain | Background execution identity persistence (DocumentStore dual-read) |
| Severity (audit) | MAJOR — fail-closed gap |
| Protected fingerprints | Unchanged |

## Audit Finding Reference

`DocumentStoreBackgroundExecutionIdentityPersistence.load()` classified v2-partition records without `execution_id` as `legacy_v1` via shape inference, and compared conflicting v1 sources using `task_id` only. A v2-partition legacy-shaped record plus a v1-partition record with the same `task_id` but different `run_id` or `attempt_id` could evade full conflict detection.

## Root Cause

Partition semantics were not enforced at decode time; `persisted_identity_from_document_record_data()` inferred version from field presence. Document load reused that inference for both partitions and applied a partial v1-on-v1 conflict check.

## Hardening Design

- Partition-aware decoders: `decode_document_identity_v2_record`, `decode_document_identity_v1_record`.
- DocumentStore load uses decoders per partition; reconciliation unchanged for provider-neutral rules.
- Reusable `same_identity_triplet` / `same_identity_quadruplet` for comparisons.
- KV path unchanged (no partition; line-count encoding already versioned).

## Partition Version Semantics

| Partition | Expected shape | On mismatch |
|-----------|----------------|-------------|
| `intergrax.bg_exec_identity.v2:<tenant>` | `task_id`, `run_id`, `attempt_id`, `execution_id` | `InvalidBackgroundExecutionIdentityV2RecordError` |
| `intergrax.bg_exec_identity.v1:<tenant>` | `task_id`, `run_id`, `attempt_id` (no `execution_id`) | `InvalidBackgroundExecutionIdentityV1RecordError` if v2-shaped |

## Conflict Semantics

- Legacy vs legacy (defensive): full triplet must match or `BackgroundExecutionIdentityConflictError`.
- v1 vs v2: triplet must match or `BackgroundExecutionIdentityConflictError` (in `reconcile_dual_read_records`).
- No heuristic preference on conflict; version precedence only when records agree.

## Typed Errors

| Problem | Error |
|---------|-------|
| v2 partition legacy shape | `InvalidBackgroundExecutionIdentityV2RecordError` |
| v1 partition v2 shape | `InvalidBackgroundExecutionIdentityV1RecordError` |
| v1/v2 triplet mismatch | `BackgroundExecutionIdentityConflictError` |
| v2 quadruplet mismatch (semantic helper) | `same_identity_quadruplet` → false; reconcile single v2 row |

Both new errors extend `Npsc5fCompatibilityError` (defined in `identity_record_codec.py`).

## Reused Components

- `validate_task_id`, `validate_run_id`, `validate_attempt_id`, `validate_execution_id` from `intergrax.contracts.execution_identity`.
- `reconcile_dual_read_records` in `identity_dual_read.py`.
- Existing NPSC-5F compatibility error types for conflict and legacy paths.

## Tests

- `tests/unit/runtime/background_execution/test_identity_persistence_conflict_hardening.py` — cases 1–8, audit regressions, preserved v1-only fail-closed behavior.
- `tests/unit/runtime/observability/test_npsc5f_v1_v2_compatibility.py` — regression.
- `tests/unit/runtime/background_execution/test_background_execution_identity.py` — regression.

## Regression Evidence

Session logs: `.tmp/session/npsc5f-persistence-hardening/pytest-identity.log`, `pytest-regression.log`.

| Suite | Result |
|-------|--------|
| Identity + compatibility unit | 51 passed |
| NPSC-5F P0 + R1 + identity single-authority gate | 32 passed |

## Static Quality

| Check | Result |
|-------|--------|
| `ruff check` (changed scope) | PASS |
| `ruff format --check` (changed scope) | PASS (after format) |
| `pyright` (changed typed scope) | Pre-existing `wire_background_execution_identity_persistence` DocumentStore protocol note only |
| `git diff --check` | PASS |

## Production Changes

| File | Change |
|------|--------|
| `identity_record_codec.py` | Partition decoders, typed errors, triplet/quadruplet helpers |
| `identity_persistence.py` | DocumentStore partition-aware load, full triplet v1 conflict |
| `identity_dual_read.py` | Import shared `same_identity_triplet` |

## Findings

- KV persistence: no partition misclassification; no change required.
- Audit regression scenarios now fail closed deterministically.

## Commit

`INTEGRAx-NPSC-5F-V1-V2-PERSISTENCE-CONFLICT-HARDENING` on `development` (single commit; SHA in session report).

## Final Verdict

**NPSC-5F V1/V2 PERSISTENCE CONFLICT HARDENING = PASS**
