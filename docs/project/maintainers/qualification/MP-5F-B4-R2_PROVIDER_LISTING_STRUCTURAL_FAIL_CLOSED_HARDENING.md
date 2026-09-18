# MP-5F-B4-R2 — Provider Listing Structural Fail-Closed Hardening

## 1. Scope

Close the last public trust-boundary gap after MP-5F-B4-R1: malformed or semantically invalid `CollaborativeWorkScopedReferenceListing` rows from custom catalogs must map to `UNAVAILABLE` / `catalog_contract_violation` without escaping `CollaborativeWorkReferenceReadScopeError` or other projection errors.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B4_R2_SESSION_START_HEAD` | `8774a27ab057defda903e09b4d67d23aa82a2408` |
| Branch | `development` |
| Session start `HEAD == origin/development` | **yes** |

## 3. Original failure mode

A custom catalog could return a `work_item` listing with `work_item_state="open"` (runtime `str`). `CollaborativeWorkScopedReferenceListing.__post_init__` did not validate enum type. `_to_canonical_ref` built `CollaborativeWorkItemCanonicalRef`, whose `__post_init__` raised **`CollaborativeWorkReferenceReadScopeError`**, which was **not** caught by the reader (only `CollaborativeWorkReferenceReadConfigurationError` was handled).

## 4. Public trust boundary

`DefaultCollaborativeWorkReferenceReader.read_references` remains the sole defensive boundary. Provider output is processed only through narrow listing validation + canonical projection; violations never propagate as uncaught exceptions.

## 5. DTO-owned validation

`CollaborativeWorkScopedReferenceListing.__post_init__` enforces entity-kind matrix (IDs, forbidden fields) and, for `work_item`, requires `work_item_state` to be **`WorkItemState`** (`isinstance`).

## 6. Reader-owned validation

- Listing vs `CollaborativeWorkScopedReferenceQuery` (tenant, workspace, entity kinds, resource filters).
- Post-catalog runtime semantics via `_validate_provider_listing_semantics` (catches bypass mutations).
- Limit cardinality (`len(listings) > limit`).

## 7. Exception classification

Caught **only** inside the per-listing projection loop (`_project_provider_listing`):

- `CollaborativeWorkReferenceReadConfigurationError`
- `CollaborativeWorkReferenceReadScopeError`
- `ValueError`
- `TypeError`

**Not** caught around the full reader. `catalog.list_scoped_references` still maps unexpected execution failures to `backend_error`.

## 8. Fail-closed mapping

Any provider listing contract violation in the batch → `outcome=UNAVAILABLE`, `reason=catalog_contract_violation`, `references=()` (no partial subset).

## 9. Malformed work-item state proof

`test_provider_malformed_work_item_state_fail_closed` — catalog mutates `work_item_state` to `"open"` after construction → `catalog_contract_violation`, no escape.

## 10. Structural malformed listing proof

`test_provider_malformed_work_artifact_listing_fail_closed` — catalog clears `current_version_id` on `work_artifact` row → `catalog_contract_violation`.

## 11. Existing R1 protections

Entity kind, limit, tenant/workspace, work_item / artifact / version filters, inconsistent parent chain — unchanged fail-closed behavior (B4 gate).

## 12. CURRENT_ONLY ownership

Unchanged: **catalog** owns `include_historical` / CURRENT_ONLY projection; reader does not verify authoritative `current_version_id` without aggregate evidence.

## 13. Canonical WorkArtifactVersion reference

**`WorkArtifactVersionRef`** remains the sole canonical version locator. **`CollaborativeWorkArtifactVersionCanonicalRef`** remains absent (R1 decision).

## 14. Pluginability

Custom `CollaborativeWorkScopedReferenceCatalog` and `CollaborativeWorkReferenceReadPort` static contracts retained in B4 gate; no concrete repository dependency in reader.

## 15. B5 compatibility

B5 adapter and ContextView port regression suites green; no B5 production edits.

## 16. Static typing

```bash
uv run pyright \
  intergrax/collaborative_work/default_collaborative_work_reference_reader.py \
  tests/unit/collaborative_work/test_cw_mp5f_b4_collaborative_work_reference_read.py
```

**0 errors, 0 warnings** on modified reader + B4 test module.

## 17. Regression evidence

```bash
uv run pytest \
  tests/unit/collaborative_work/test_cw_mp5f_b4_collaborative_work_reference_read.py \
  tests/unit/collaborative_work/test_mp5f_b5_context_view_source_adapters.py \
  tests/unit/contracts/test_context_view_source_ports.py \
  tests/unit/collaborative_work/test_context_view_source_ports_architecture_gates.py \
  -q
```

**65 passed**.

## 18. Final B4 verdict

All known provider-listing contract violations fail closed at the public reader — B4 eligible for **CLOSED / CERTIFIED** pending independent audit.
