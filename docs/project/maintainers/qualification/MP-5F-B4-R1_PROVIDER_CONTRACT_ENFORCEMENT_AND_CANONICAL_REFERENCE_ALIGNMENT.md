# MP-5F-B4-R1 — Provider Contract Enforcement & Canonical Reference Alignment

## 1. Scope

Close enterprise gaps after MP-5F-B4: defensive provider contract enforcement on `DefaultCollaborativeWorkReferenceReader` and single canonical `WorkArtifactVersion` locator authority (`WorkArtifactVersionRef`).

## 2. Git provenance

Recorded at implementation time on branch `development` (see session commit message for SHAs).

## 3. Original findings

- Reader validated tenant/workspace/partial resource scope but not entity kinds ⊆ query, limit cardinality, or strict optional artifact/version filters.
- `CollaborativeWorkArtifactVersionCanonicalRef` duplicated `WorkArtifactVersionRef` identity tuple.

## 4. Provider trust boundary

`DefaultCollaborativeWorkReferenceReader` is the **public defensive boundary**. `CollaborativeWorkScopedReferenceCatalog` implementations are **not** blindly trusted for scope, kinds, or limit.

## 5. Reader-owned invariants

- Request identity vs scope tenant alignment (via `validate_collaborative_work_reference_read_request`).
- Capability binding tenant/workspace.
- Listing tenant/workspace vs query.
- Listing `entity_kind ∈ query.entity_kinds`.
- Optional `work_item_id`, `work_artifact_id`, `work_artifact_version_id` filters vs listing semantics.
- `len(listings) <= query.limit`.
- Canonical projection from structurally valid listings (DTO `__post_init__` for row shape).
- Deterministic sort after validation.

## 6. Catalog-owned invariants

- `include_historical` / CURRENT_ONLY projection using aggregate `current_version_id` (not timestamps).
- Scope filtering performance path before limit.
- Deterministic ordering before limit slice.
- Structurally valid listings returned to reader.

## 7. Entity-kind enforcement

Unrequested `entity_kind` → `UNAVAILABLE` / `catalog_contract_violation`.

## 8. Limit enforcement

`len(listings) > query.limit` → `UNAVAILABLE` / `catalog_contract_violation` (no silent truncation). Default repository catalog still applies limit after filtering (defense in depth).

## 9. Resource scope enforcement

Strict filters: when query specifies artifact or version id, artifact/version listings must match; work_item mismatches fail closed.

## 10. Version-selection ownership

**CATALOG** owns CURRENT_ONLY vs INCLUDE_HISTORICAL. Reader documents this and does not heuristic-verify current pointer from a version row alone.

## 11. `WorkArtifactVersionRef` comparison

| Property | WorkArtifactVersionRef | Former B4 Canonical Ref |
| --- | --- | --- |
| tenant | yes | yes |
| workspace | yes | yes |
| work item | yes | yes |
| artifact | yes | yes |
| version | yes | yes |
| schema version | `work_artifact_version_ref.v1` | none |
| intended owner | Collaborative Work domain contract | B4 read projection (removed) |
| lifecycle semantics | immutable locator | same |
| public consumers | Evidence, ContextView, B5 | B4 only (removed) |

## 12. Canonical reference decision

**`WorkArtifactVersionRef`** is the sole canonical WorkArtifactVersion locator. B4 `CollaborativeWorkCanonicalRef` union includes `WorkArtifactVersionRef` directly; duplicate dataclass removed.

## 13. Migration/compatibility impact

B5 mapping accepts `WorkArtifactVersionRef` from B4 read results (identity pass-through). No ContextView port change.

## 14. Pluginability

Custom `CollaborativeWorkReferenceReadPort` and `CollaborativeWorkScopedReferenceCatalog` remain supported without concrete repository imports.

## 15. Backend parity

`RepositoryBackedCollaborativeWorkReferenceCatalog` unchanged semantically; InMemory/SQLite paths covered by existing B4 tests.

## 16. B5 compatibility

Mapping updated minimally; B5 still repository-independent.

## 17. Static typing

Pyright clean on modified production and B4 test module (see regression evidence).

## 18. Regression evidence

`uv run pytest` on B4 + B5 + ContextView port/architecture gates; `uv run pyright` on listed modules.

## 19. Final B4 verdict

Provider contract hardening and canonical version locator alignment complete — B4 eligible for **CLOSED / CERTIFIED** pending independent audit.
