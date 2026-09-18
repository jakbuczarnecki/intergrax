# MP-5F-B3A-C1 — Production Ownership Wiring Recertification

## 1. Scope

Recertification of the production path:

```text
ContextAssemblyRequest
→ ContextEngine (DefaultNexusContextEngine)
→ resolve_ucl_artifact_ownership_scope(...)
→ resolve_ucl_context_plan(..., artifact_ownership=...)
→ OptimizationArtifactRepository (lookup / reservation / persist / reuse)
```

Audited modules (HEAD at certification):

- `intergrax/context/contracts.py` (`ContextAssemblyRequest.workspace_id`)
- `intergrax/runtime/nexus/context/context_engine.py`
- `intergrax/runtime/nexus/context/ucl_artifact_ownership_composition.py`
- `intergrax/runtime/nexus/context/ucl_orchestration.py`
- `intergrax/runtime/context_lifecycle/contracts.py`
- `intergrax/runtime/context_lifecycle/repository.py`
- `intergrax/runtime/context_lifecycle/in_memory_repository.py`
- `intergrax/runtime/context_lifecycle/sqlite_repository.py`

Out of scope: B3B reader, Collaborative Work read API, MP-5D adapters, ContextView integration, repository redesign.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B3A_C1_SESSION_START_HEAD` | `5f9ada7a64e378fa58f038bb47d7b2ec0cc0bf48` |
| Branch | `development` |
| `HEAD == origin/development` at session start | yes |

Relevant prior lineage (inspected, audit against current HEAD only): workspace ownership composition and repository partition keys landed after C0 design (`4cd345a15…`, `a0c0d2be…`).

## 3. Canonical ownership model

| Dimension | Owner / contract | Notes |
| --- | --- | --- |
| Tenant ownership | `UclArtifactOwnershipScope.tenant_id` | Must match `ArtifactLookupKey.tenant_id` before repository use; mismatch → `PLAN_MATERIALIZATION_FAILED`. |
| Workspace ownership | `UclArtifactOwnershipScope.workspace_id` | Partition key component `(tenant, workspace, lookup_hash)`; not derived from `context_scope_id`. |
| Context scope | `ArtifactLookupKey.context_scope_id` / session history | Lifecycle and lookup identity only; orthogonal to workspace. |
| Artifact lookup identity | `ArtifactLookupKey` + `compute_artifact_lookup_key_hash` | Content/strategy/policy identity; workspace is **not** mixed into hash (ownership is separate). |

`UclArtifactOwnershipScope` is a frozen dataclass in `context_lifecycle.contracts` — implementation-neutral, no CW/Nexus private types.

## 4. Composition path

`resolve_ucl_artifact_ownership_scope(request, context_plan)`:

| Condition | Result |
| --- | --- |
| `optimization_required` false | `None` |
| `optimization_required` true + non-empty `request.workspace_id` | `UclArtifactOwnershipScope(tenant_id=request.tenant_id, workspace_id=stripped)` |
| `optimization_required` true + missing/blank workspace | `None` → UCL fails closed |

Single composition entry from `ContextEngine`; sources are only `request.tenant_id` and `request.workspace_id`.

## 5. Repository contract

`OptimizationArtifactRepository` requires `ownership: UclArtifactOwnershipScope` on `lookup`, `try_acquire_creation_reservation`, and `wait_for_artifact_or_reservation_change`. Partition via `partition_key_for_ownership_scope`. Persisted artifacts carry `UclArtifactOwnership.for_workspace(scope)`.

## 6. Create path

`CREATE_ARTIFACT`: reservation acquired with ownership scope → executor → `store_validated_artifact` with matching `UclArtifactOwnership.for_workspace(artifact_ownership)` → subsequent `lookup` uses same scope.

## 7. Reuse path

`REUSE_ARTIFACT`: `lookup(key, ownership=...)` only returns artifacts in the ownership partition; stored payload validation does not bypass scope.

## 8. Reservation path

Reservations keyed by `(tenant_id, workspace_id, lookup_hash)`; identical lookup keys in different workspaces do not share reservation state (behavioral tests T7).

## 9. Same-workspace behavior

Same tenant + workspace + lookup identity → reuse allowed (T3, T6, T10).

## 10. Cross-workspace isolation

Same tenant, different workspace, identical lookup dimensions → miss / no reuse (T4). Behavioral proof on InMemory and SQLite backends.

## 11. Cross-tenant isolation

Different tenant, same workspace string and lookup dimensions → miss (T5). `resolve(reference)` enforces tenant on reference (existing in-memory tests).

## 12. Backend parity

Parametrized tests (`memory`, `sqlite`) in `test_b3a_workspace_ownership.py` cover isolation, reservation independence, reuse, and context_scope ≠ workspace_id.

## 13. Pluginability / replaceability

`ucl_orchestration` depends only on `OptimizationArtifactRepository` protocol via `NexusUCLRuntimeDependencies`; no import of concrete SQLite/InMemory types in orchestration.

## 14. Boundary integrity

- No CW imports in composition/orchestration wiring gates.
- No second ownership DTO; no metadata/session fallback for workspace.
- C1 hardening: ownership `None` / tenant mismatch validated immediately after strategy selection, **before** `_prepare_artifact_materialization` and any repository call.

## 15. Findings

| ID | Class | Severity | Status |
| --- | --- | --- | --- |
| F1 | Test drift (`lookup`/`reservation` without `ownership` in orchestration tests) | low | **fixed** in C1 commit |
| F2 | Fail-closed ordering (ownership after heavy prep) | low | **fixed** — early guard in `resolve_ucl_context_plan` |
| — | `resolve(reference)` without separate ownership arg | info | **accepted** — reference carries `tenant_id` + `workspace_id`; cross-tenant `resolve` returns `None` (not on hot UCL optimization path) |

No OPEN: OWNERSHIP-MISSING, CROSS-WORKSPACE-LEAK, CROSS-TENANT-LEAK, RESERVATION-SCOPE-GAP, BACKEND-PARITY-GAP, DUPLICATE-OWNERSHIP-MECHANISM, INTERNAL-LEAK, ADR.

## 16. Test evidence

Command (session log: `.tmp/session/MP-5F-B3A-C1/pytest-regression2.log`):

```bash
uv run pytest \
  tests/unit/runtime/context_lifecycle/test_b3a_workspace_ownership.py \
  tests/unit/runtime/nexus/context/test_ucl_orchestration.py \
  tests/unit/runtime/nexus/context/test_b3a_c1_context_engine_ownership_wiring.py \
  tests/unit/runtime/architecture/test_ucl_b3a_workspace_ownership_gate.py \
  tests/unit/runtime/context_lifecycle/test_repository_contracts.py \
  tests/unit/runtime/context_lifecycle/test_in_memory_repository.py \
  tests/unit/runtime/context_lifecycle/test_sqlite_repository.py \
  -q
```

Result: **150 passed**.

Matrix T1–T10:

| Test | Coverage |
| --- | --- |
| T1 | `test_engine_selection_only_without_workspace_succeeds` |
| T2 | `test_missing_artifact_ownership_fails_before_repository` |
| T3 | `test_same_workspace_reuse_allowed` |
| T4 | `test_cross_workspace_lookup_miss_with_same_key` |
| T5 | `test_cross_tenant_same_workspace_string_isolated` |
| T6 | create + lookup via orchestration / repository publish helpers |
| T7 | `test_cross_workspace_reservation_isolated` |
| T8 | `test_concurrent_same_key_single_flight` (orchestration) |
| T9 | `test_artifact_available_retries_lookup_without_executor` |
| T10 | `test_context_scope_may_differ_from_workspace` / composition test |

## 17. Certification verdict

**CLOSED / CERTIFIED** — ownership reaches all required repository operations; tenant+workspace isolation behaviorally proven on both default backends; fail-closed before repository and executor when ownership is missing.

## 18. B3B readiness

**YES** — MP-5F-B3B can rely on workspace-scoped UCL ownership at the repository boundary without introducing a second ownership mechanism.
