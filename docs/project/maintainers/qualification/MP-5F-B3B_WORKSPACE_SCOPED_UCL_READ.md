# MP-5F-B3B — Workspace-Scoped UCL Read

## 1. Scope

Public, reference-only UCL read for optimization artifacts scoped by:

```text
tenant_id + workspace_id + context_scope_id
```

In scope: `OptimizationArtifactScopedReferenceQuery`, `OptimizationArtifactScopedReferenceCatalog`, `ScopedOptimizationArtifactListing`, `DefaultUclReferenceReader`, `UclReferenceReadPort` / `UclReferenceReadScope`, InMemory and SQLite catalog implementations.

Out of scope: payload hydration, ContextView adapter (MP-5D/B5), Collaborative Work read (B4), Nexus-owned read contracts, repository redesign, new ownership or reference DTOs.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B3B_SESSION_START_HEAD` | `ea342b2d9c906ac59cf56acc7c7df67d76fbd412` |
| `MP5F_B3B_EVIDENCE_HEAD` | `ea342b2d9c906ac59cf56acc7c7df67d76fbd412` (pre-certification closeout) |
| Branch | `development` |
| `HEAD == origin/development` at session start | **yes** |
| Implementation commits (pre-doc) | `38f10f97c` (workspace-scoped reference read), `ea342b2d9` (resource scope before limit) |

Audit performed on task branch at certification closeout.

## 3. Canonical read owner

Semantic owner: **UCL / context lifecycle** (`intergrax/runtime/context_lifecycle/repository.py`, `default_ucl_reference_reader.py`).

Application-facing port: **`intergrax/ucl/contracts/ucl_reference_read.py`** (`UclReferenceReadPort`).

Not owned by Nexus, CE, SQLite adapter surface, ContextView, or Collaborative Work.

## 4. Public query contract

```text
OptimizationArtifactScopedReferenceQuery
  tenant_id: str          (required, non-empty)
  workspace_id: str       (required, non-empty)
  context_scope_id: str   (required, non-empty)
  limit: int              (required, > 0)
  include_historical: bool = False
  source_ref: str | None = None   (optional catalog filter; applied before limit)
```

Validation in `__post_init__`: empty scope fields and `limit <= 0` → `ValueError`.

## 5. Public result contract

**Catalog row:** `ScopedOptimizationArtifactListing` — `OptimizationArtifactReference`, `context_scope_id`, `lifecycle_status`, `source_refs`. No payload.

**Reader result:** `UclReferenceReadResult` with `UclOptimizationArtifactCanonicalRef` tuples (reference-only projection).

**Reference model:** canonical `OptimizationArtifactReference` via `build_optimization_artifact_reference(stored)` from persisted `metadata.ownership`.

## 6. Workspace enforcement

- Query requires explicit `workspace_id` (not optional, not substituted by `context_scope_id`).
- InMemory: ownership filter on `metadata.ownership.scope.workspace_id` before emit.
- SQLite: `WHERE workspace_id = ?` with `ownership_kind = WORKSPACE`.
- `LEGACY_UNKNOWN` and non-WORKSPACE kinds excluded from workspace-scoped catalog paths.
- `DefaultUclReferenceReader` rejects request scope mismatch vs capability binding (fail closed).

## 7. Tenant isolation

Catalog matches `ownership_scope.tenant_id` and lookup key `tenant_id` (InMemory) or SQL `tenant_id = ?` (SQLite). Cross-tenant rows cannot appear for a scoped query.

## 8. Context scope isolation

Independent dimension: `lookup_key.context_scope_id` / JSON extract must equal query `context_scope_id`. `workspace_id != context_scope_id` supported (e.g. workspace `ws-a`, context `ctx-77`).

## 9. Historical semantics

- `include_historical=False`: active eligible validated entries only (`VALIDATED` + validation `PASSED`; InMemory active index / SQLite status predicates).
- `include_historical=True`: historical lifecycle rows included; still filtered by tenant, workspace, context scope, and WORKSPACE ownership.

## 10. Deterministic ordering

Canonical sort: `(artifact_id, artifact_lookup_key_hash)`.

- InMemory: sort then `[:limit]`.
- SQLite: `ORDER BY artifact_id, lookup_key_hash` then cap at `limit`.

Backend parity proven in `test_in_memory_sqlite_parity_same_order` (B3B-C1 suite).

## 11. Backend parity

Parametrized behavioral tests (`memory`, `sqlite`) in:

- `tests/unit/ucl/test_ucl_mp5f_b3b_workspace_scoped_reference_read.py`
- `tests/unit/ucl/test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py`

Same query → same ordering and scoped reference sets.

## 12. Pluginability

- Read consumers may depend on `OptimizationArtifactScopedReferenceCatalog` (narrow) without full `OptimizationArtifactRepository` mutation surface.
- `DefaultUclReferenceReader` accepts any catalog implementing the protocol (`test_custom_catalog_receives_workspace_query`).
- `UclReferenceReadPort` allows fully custom async implementations (`test_custom_read_port_still_supported`).
- Concrete InMemory/SQLite classes remain replaceable; UCL semantics live in contracts, not providers.

## 13. Reference-only guarantee

`list_scoped_artifact_references` does not call `resolve()` for hydration. Results are references and lifecycle projection only — no `StoredOptimizationArtifact`, payload bytes, row ids, or storage paths in public listing/reader types.

## 14. Source-port readiness

**YES.** A future MP-5D / ContextView adapter can consume `UclReferenceReadPort` or `OptimizationArtifactScopedReferenceCatalog` plus `UclOptimizationArtifactCanonicalRef` without importing SQLite/InMemory or Nexus private types. UCL B3 read is async; MP-5D source ports are sync — B5 integration must bridge explicitly (documented in architecture hub).

## 15. Findings

| ID | Severity | Finding | Status |
| --- | --- | --- | --- |
| F1 | — | Query already includes mandatory `workspace_id` on HEAD | **closed** |
| F2 | — | SQLite scopes workspace in SQL predicate (no post-filter leak) | **closed** |
| F3 | — | Scoped catalog types not exported from `context_lifecycle` package | **remediated** (public `__init__` export) |
| F4 | — | Qualification artifact missing | **remediated** (this document) |
| F5 | info | Authorization (principal ↔ workspace) remains upstream; UCL enforces storage ownership only | **accepted** |

## 16. Tests

Regression (session):

```text
uv run pytest \
  tests/unit/runtime/context_lifecycle/test_b3a_workspace_ownership.py \
  tests/unit/runtime/context_lifecycle/test_b3a_c1r1_reference_resolution_hardening.py \
  tests/unit/runtime/context_lifecycle/test_repository_contracts.py \
  tests/unit/runtime/context_lifecycle/test_in_memory_repository.py \
  tests/unit/runtime/context_lifecycle/test_sqlite_repository.py \
  tests/unit/ucl/test_ucl_mp5f_b3b_workspace_scoped_reference_read.py \
  tests/unit/ucl/test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py \
  -q
```

Result: **162 passed**.

Coverage mapping (T1–T10): workspace listing, cross-workspace/tenant/context isolation, binding rejection, historical modes, limit-after-scope, resource `source_ref` catalog scope (B3B-C1), custom catalog protocol, anti-regression (no workspace==context rule in reader).

## 17. Certification verdict

**MP-5F-B3B — CLOSED / CERTIFIED**

## 18. MP-5F-B3 readiness

**YES** — workspace-scoped reference read boundary is implemented, tested, and documented. MP-5F-B3 can proceed to close the full UCL read boundary toward ContextView integration (adapter in B5, not B3B).
