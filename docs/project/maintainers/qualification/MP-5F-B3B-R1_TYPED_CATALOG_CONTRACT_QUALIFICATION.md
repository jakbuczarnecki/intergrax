# MP-5F-B3B-R1 — Typed Catalog Contract Qualification

## 1. Scope

Close type-safety and plugin-contract qualification gaps after MP-5F-B3B without production redesign.

In scope: B3B qualification tests (`test_ucl_mp5f_b3b_workspace_scoped_reference_read.py`, `test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py`), static assignability of custom `OptimizationArtifactScopedReferenceCatalog` and `UclReferenceReadPort`, qualification artifact.

Out of scope: repository redesign, B4, ContextView adapter, new read ports, ownership DTOs, backend coupling in consumer contracts.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B3B_R1_SESSION_START_HEAD` | `9b8b826810000ac9db041546636c41228540a859` |
| `MP5F_B3B_R1_EVIDENCE_HEAD` | `9b8b826810000ac9db041546636c41228540a859` |
| Branch | `development` |
| `HEAD == origin/development` at session start | **yes** |

## 3. Prior gaps

- `ArtifactLookupKey` test builder used `dict[str, object]` + `**defaults` and `# type: ignore[arg-type]` in both B3B test modules.
- `_CustomCatalog` in workspace B3B tests declared `tuple[object, ...]` instead of `tuple[ScopedOptimizationArtifactListing, ...]`.
- Custom catalog plugin proof relied on runtime duck typing without explicit `OptimizationArtifactScopedReferenceCatalog` assignment.
- No backend-independent test returning a non-empty `ScopedOptimizationArtifactListing` through `DefaultUclReferenceReader`.

## 4. Type-safety cleanup

- Replaced `_lookup_key` with explicit keyword parameters and direct `ArtifactLookupKey(...)` construction (no `**overrides`, no `type: ignore`).
- Updated `_stored` helpers to call `_lookup_key` with typed arguments only.

## 5. Catalog protocol shape

Canonical protocol (`intergrax/runtime/context_lifecycle/repository.py`):

```python
def list_scoped_artifact_references(
    self,
    query: OptimizationArtifactScopedReferenceQuery,
) -> tuple[ScopedOptimizationArtifactListing, ...]:
```

`DefaultUclReferenceReader.catalog` is typed as `OptimizationArtifactScopedReferenceCatalog | None`.

## 6. Custom catalog static conformance

Test implementations:

```python
def list_scoped_artifact_references(
    self,
    query: OptimizationArtifactScopedReferenceQuery,
) -> tuple[ScopedOptimizationArtifactListing, ...]:
```

Explicit assignment without cast/ignore/Any:

```python
catalog: OptimizationArtifactScopedReferenceCatalog = _CustomCatalog()
```

## 7. Custom catalog behavioral proof

- Workspace B3B: asserts `tenant_id`, `workspace_id`, `context_scope_id` on catalog query.
- C1 B3B: `_RecordingCatalog` asserts full query (`tenant_id`, `workspace_id`, `context_scope_id`, `source_ref`, `limit`, `include_historical`).
- Non-empty path: `_CustomCatalogWithListing` returns `ScopedOptimizationArtifactListing`; reader emits `UclOptimizationArtifactCanonicalRef` (reference-only).

## 8. Custom read-port conformance

```python
port: UclReferenceReadPort = _CustomReadPort()
```

No `Any`, cast, or dynamic adapter.

## 9. Backend independence

Custom catalog and read-port qualification tests do not instantiate `InMemoryOptimizationArtifactRepository` or `SQLiteOptimizationArtifactRepository`. Backend parametrized tests remain separate.

## 10. Production code impact

**None.** Contract shape at HEAD already matched qualification requirements; fixes are test-only.

## 11. Regression evidence

```bash
uv run pytest \
  tests/unit/ucl/test_ucl_mp5f_b3b_workspace_scoped_reference_read.py \
  tests/unit/ucl/test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py \
  tests/unit/runtime/context_lifecycle/test_b3a_workspace_ownership.py \
  tests/unit/runtime/context_lifecycle/test_b3a_c1r1_reference_resolution_hardening.py \
  tests/unit/runtime/context_lifecycle/test_repository_contracts.py \
  tests/unit/runtime/context_lifecycle/test_in_memory_repository.py \
  tests/unit/runtime/context_lifecycle/test_sqlite_repository.py \
  -q
```

Result: **163 passed**.

Static check (modified B3B tests):

```bash
uv run pyright \
  tests/unit/ucl/test_ucl_mp5f_b3b_workspace_scoped_reference_read.py \
  tests/unit/ucl/test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py
```

Custom catalog assignability lines: **0 errors**. Pre-existing pyright notes on passing full `OptimizationArtifactRepository` into `_reader` helpers (runtime structural match via `list_scoped_artifact_references` on backends) unchanged by R1.

```bash
git diff --check
```

## 12. Final B3B certification

| Contract | Static conformant | Runtime | Backend-independent |
| --- | ---: | ---: | ---: |
| `OptimizationArtifactScopedReferenceCatalog` | YES | PASS | YES |
| `UclReferenceReadPort` | YES | PASS | YES |

MP-5F-B3B may be upgraded to **CLOSED / CERTIFIED** after this commit on `development`.

## 13. MP-5F-B3 readiness

UCL read boundary (B3) can proceed toward formal closure: reference-only scoped read, typed catalog contract, and plugin qualification are satisfied at test/certification layer; production semantics unchanged.
