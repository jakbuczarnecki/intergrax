# MP-5F-B4-R2-R1 — Immutable Provider Qualification Cleanup

## 1. Scope

Remove forbidden immutable-DTO mutation bypasses from the MP-5F-B4 qualification gate while preserving MP-5F-B4-R2 production fail-closed semantics. Qualification evidence is split across DTO construction, reader/query contract checks, and legal custom-catalog behavioral proofs.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B4_R2_R1_SESSION_START_HEAD` | `4b5f148b6f04ffed7482039822f5a889908cf260` |
| `MP5F_B4_R2_R1_EVIDENCE_HEAD` | `4b5f148b6f04ffed7482039822f5a889908cf260` |
| Branch | `development` |
| Session start `HEAD == origin/development` | **yes** |

Historical R2 context (not rewritten):

| Field | Value |
| --- | --- |
| `MP5F_B4_R2_SESSION_START_HEAD` | `8774a27ab057defda903e09b4d67d23aa82a2408` |
| Parallel branch drift before R2 commit | **yes** — `development` advanced between R2 session start and R2 land |
| `MP5F_B4_R2_PRE_PARALLEL_EVIDENCE_HEAD` | `8774a27ab057defda903e09b4d67d23aa82a2408` (R2 doc session marker; not the direct parent of R2) |
| `MP5F_B4_R2_ACTUAL_TASK_PARENT` | `d0aee0668` (MP-5H final certification commit) |
| `MP5F_B4_R2_FINAL_TASK_HEAD` | `38c5baf83` (`fix(collaborative-work): fail closed on malformed catalog listings`) |

R2-R1 task commit parent is the current `development` tip at evidence time (after parallel commits `38c5baf83` → … → session start).

## 3. Original qualification defect

B4 gate tests used `object.__setattr__` on frozen `CollaborativeWorkScopedReferenceListing` to simulate post-construction corruption:

- `object.__setattr__(listing, "work_item_state", "open")`
- `object.__setattr__(listing, "current_version_id", None)`

This violated platform qualification standards (no mutation of frozen DTOs in qualification).

## 4. Immutable DTO rule

Qualification must not bypass `frozen=True` on platform contracts. Structural invalidity is proven at **construction** or through **valid DTOs** that violate the **query/reader** contract.

## 5. Removed mutation bypasses

Deleted `_MalformedWorkItemStateCatalog`, `_MalformedWorkArtifactShapeCatalog`, and reader tests that depended on post-construction mutation. **No replacement bypass** (no `setattr`, reflection, casts, or typing suppressions).

## 6. DTO-owned structural validation

| Case | Proof |
| --- | --- |
| `work_artifact` without `current_version_id` | `test_listing_work_artifact_missing_current_version_id_rejected_at_construction` → `ValueError` at construction |
| `work_item` without `WorkItemState` | `test_listing_work_item_missing_work_item_state_rejected_at_construction` → `ValueError` at construction |

**Invalid `work_item_state` as raw `str`:** static typing prevents illegal construction in qualification without a forbidden bypass. Runtime defense-in-depth remains in `CollaborativeWorkScopedReferenceListing.__post_init__` (`isinstance(..., WorkItemState)`). Post-construction memory corruption is out of scope for B4 functional qualification.

## 7. Reader-owned query validation

Custom catalogs return **valid** listings; violations are query-scope mismatches (tenant, workspace, work item, artifact, version, entity kind, limit, parent chain). Proven by existing B4 gate catalog fixtures (`test_plugin_*` / wrong-* fail-closed matrix).

## 8. Legal provider contract tests

Projection mapping to `UNAVAILABLE` / `catalog_contract_violation` for out-of-scope listings uses valid DTOs (`test_plugin_out_of_scope_listing_fail_closed`). Backend execution failures remain `backend_error`.

## 9. Production impact

**production code changed: NO** — R2 reader and DTO validation unchanged.

## 10. Canonical reference freeze

**`WorkArtifactVersionRef`** remains the sole canonical WorkArtifactVersion locator. **`CollaborativeWorkArtifactVersionCanonicalRef`** remains absent.

## 11. CURRENT_ONLY ownership freeze

**`CollaborativeWorkScopedReferenceCatalog`** owns `include_historical` / CURRENT_ONLY projection; reader unchanged.

## 12. Pluginability

| Check | Result |
| --- | --- |
| Custom catalog (`CollaborativeWorkScopedReferenceCatalog`) | **PASS** (`test_external_catalog_without_concrete_repository`) |
| Custom port (`CollaborativeWorkReferenceReadPort`) | **PASS** (`test_custom_port_implementation`) |
| Concrete repository in default reader | **NONE** (import gates) |

## 13. B5 compatibility

B5 regression suite included in gate; **no B5 production edits** in R2-R1.

## 14. Escape-hatch audit

B4 qualification test module: **zero** `object.__setattr__`, `setattr`, `getattr`, `hasattr`, `__dict__`, `Any`, `cast`, `type: ignore`, `pyright: ignore`, `noqa`, `inspect` (AST import gates only).

## 15. Pyright

```bash
uv run pyright \
  intergrax/collaborative_work/default_collaborative_work_reference_reader.py \
  tests/unit/collaborative_work/test_cw_mp5f_b4_collaborative_work_reference_read.py
```

**0 errors, 0 warnings** (recorded at R2-R1 closure).

## 16. Regression

```bash
uv run pytest \
  tests/unit/collaborative_work/test_cw_mp5f_b4_collaborative_work_reference_read.py \
  tests/unit/collaborative_work/test_mp5f_b5_context_view_source_adapters.py \
  tests/unit/contracts/test_context_view_source_ports.py \
  tests/unit/collaborative_work/test_context_view_source_ports_architecture_gates.py \
  -q
```

**65 passed** at R2-R1 closure.

## 17. R2 final status

Production trust boundary from **MP-5F-B4-R2** remains valid. R2 qualification narrative corrected; structural proofs no longer rely on immutable mutation.

## 18. B4 final status

With R2-R1 cleanup, **MP-5F-B4 — CLOSED / CERTIFIED** (pending independent audit).

## 19. MP-5H delta recertification requirement

MP-5H final certification commit **`d0aee0668`** preceded B4-R2 commit **`38c5baf83`**. Existing MP-5H certification is therefore **older than final B4 hardening**. **MP-5H delta recertification is required** after B4 closure; not performed in this task.
