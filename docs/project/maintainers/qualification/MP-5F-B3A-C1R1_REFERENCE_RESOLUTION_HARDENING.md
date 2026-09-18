# MP-5F-B3A-C1R1 — Workspace-Scoped Reference Resolution Hardening

## 1. Scope

Hardening of reference-based repository operations:

```text
resolve(reference)
invalidate_artifact(reference, ...)
retire_artifact(reference, ...)
```

Modules: `repository.py`, `in_memory_repository.py`, `sqlite_repository.py`, behavioral tests under `tests/unit/runtime/context_lifecycle/`.

Out of scope: B3B reader, ContextEngine wiring, Collaborative Work, repository redesign.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B3A_C1R1_SESSION_START_HEAD` | `1c2ebea6e72e99c1a07b632c2c49df51794a6dc5` |
| `MP5F_B3A_C1R1_EVIDENCE_HEAD` | `c90c3edcff0b168c6f60c44ce9ea6166d50685df` |
| Branch | `development` |
| `HEAD == origin/development` at session start | **no** (local HEAD behind `origin/development`) |

Audit performed on task commit only.

## 3. Root cause

**REFERENCE-WORKSPACE-GAP:** `resolve` matched `(tenant_id, artifact_id)` plus hash/type fields but did not require `reference.workspace_id` to equal canonical `metadata.ownership` workspace. Cross-workspace read/mutation was possible with a forged reference.

SQLite additionally omitted `workspace_id` from SQL predicates for resolve and lifecycle `UPDATE`.

## 4. Reference security semantics

`OptimizationArtifactReference` is a **scoped repository capability**, not authentication. Possession of `artifact_id` and hashes does not override tenant/workspace boundaries. Wrong scope → `None` (fail closed, no existence leak).

## 5. Canonical ownership matching

Shared helper `optimization_artifact_reference_matches_stored(reference, artifact)` enforces:

- `tenant_id`, `artifact_id`, `context_scope_id`
- lookup hash, content hash, `artifact_type`
- WORKSPACE kind: `reference.workspace_id` mandatory and equal to `metadata.ownership.scope.workspace_id`
- non-WORKSPACE: `reference.workspace_id` must be `None`

Workspace derived from `metadata.ownership`, not convenience projections.

## 6. InMemory implementation

`_resolve_locked` loads by `(tenant_id, artifact_id)` then applies shared reference matcher. `invalidate` / `retire` reuse `_resolve_locked` before mutation.

## 7. SQLite implementation

`_artifact_for_reference` queries with `(tenant_id, workspace_id, artifact_id)` when workspace present, else `workspace_id IS NULL`. Lifecycle `UPDATE` uses the same partition predicate after full reference validation.

## 8. Resolve behavior

Correct reference → stored envelope. Wrong workspace, tenant, or context scope → `None`.

## 9. Invalidate behavior

Validated reference → status transition. Forged workspace/tenant → `None`, artifact unchanged and still resolvable with correct reference.

## 10. Retire behavior

Same as invalidate with `RETIRED` status.

## 11. Backend parity

Parametrized tests (`memory`, `sqlite`) in `test_b3a_c1r1_reference_resolution_hardening.py` — identical outcomes.

## 12. Pluginability

Single `OptimizationArtifactRepository` protocol; helper lives in `repository.py` (provider-neutral). No second ownership DTO or backend-specific authorization.

## 13. Regression evidence

```text
uv run pytest tests/unit/runtime/context_lifecycle/... tests/unit/runtime/nexus/context/test_ucl_orchestration.py tests/unit/runtime/nexus/context/test_b3a_c1_context_engine_ownership_wiring.py -q
```

162 passed (session log: `.tmp/session/MP-5F-B3A-C1R1/pytest.log`).

## 14. Findings

| Class | Status |
| --- | --- |
| REFERENCE-WORKSPACE-GAP | **closed** |
| CROSS-WORKSPACE-READ | **closed** |
| CROSS-WORKSPACE-MUTATION | **closed** |
| BACKEND-PARITY-GAP | **closed** |
| CONTRACT-GAP | **closed** (protocol docstrings + reference DTO docs) |

## 15. Certification verdict

**MP-5F-B3A-C1R1 — CLOSED / CERTIFIED** (pending independent GitHub audit).

## 16. B3B readiness

**YES** — `OptimizationArtifactReference` and repository reference operations enforce tenant + workspace ownership consistently with lookup/reservation/create paths.
