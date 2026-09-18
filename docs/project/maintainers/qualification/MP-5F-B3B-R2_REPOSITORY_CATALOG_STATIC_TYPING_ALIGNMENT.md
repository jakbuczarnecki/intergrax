# MP-5F-B3B-R2 — Repository/Catalog Static Typing Alignment

## 1. Scope

Close the last Pyright mismatch in B3B qualification tests after MP-5F-B3B-R1: test fixtures typed only as `OptimizationArtifactRepository` were passed to `DefaultUclReferenceReader`, which requires `OptimizationArtifactScopedReferenceCatalog`.

In scope: `test_ucl_mp5f_b3b_workspace_scoped_reference_read.py`, `test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py`, this qualification artifact.

Out of scope: production reader/repository redesign, protocol merge, B4, ContextView adapter, new public contracts, backend coupling in consumer code.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B3B_R2_SESSION_START_HEAD` | `7b2b6a56b69968e4c638f8e4caef8fd55704daa1` |
| `MP5F_B3B_R2_EVIDENCE_HEAD` | `7b2b6a56b69968e4c638f8e4caef8fd55704daa1` |
| Branch | `development` |
| `HEAD == origin/development` at session start | **yes** |

## 3. Original Pyright mismatch

Pyright (pre-fix) reported **6 errors** across both B3B modules:

- `_reader` passed `OptimizationArtifactRepository` to `DefaultUclReferenceReader(catalog=...)`, which expects `OptimizationArtifactScopedReferenceCatalog | None` (`list_scoped_artifact_references` not declared on repository protocol).
- `catalog_repository` fixture annotated as returning `OptimizationArtifactRepository` while implementing a generator (`yield`); incompatible with declared return type.

Root cause: **contract segregation** — mutable repository capability ≠ scoped enumeration capability — was correct at runtime (default backends implement both) but not represented in test types.

## 4. Contract segregation model

| Role | Contract |
| --- | --- |
| Publisher / test setup | `OptimizationArtifactRepository` |
| Reader consumer | `OptimizationArtifactScopedReferenceCatalog` |
| Default in-memory / SQLite backend | implements both independently on one provider |

Repository protocol does **not** imply catalog enumeration. Catalog protocol does **not** imply mutation lifecycle. One backend object may satisfy both without merging public contracts.

## 5. Local intersection/composite type

Test-local Protocol intersection (not exported, no new platform authority):

```python
class _RepositoryWithScopedCatalog(
    OptimizationArtifactRepository,
    OptimizationArtifactScopedReferenceCatalog,
    Protocol,
):
    """Test-local intersection: mutable repository + scoped catalog enumeration."""
```

No additional methods; structural combination of two existing platform contracts only.

## 6. Fixture typing

```python
@pytest.fixture(params=("memory", "sqlite"))
def catalog_repository(...) -> Iterator[_RepositoryWithScopedCatalog]:
    ...
    yield repo
```

Parametrized tests declare `catalog_repository: _RepositoryWithScopedCatalog`.

## 7. Reader helper typing

```python
def _reader(
    catalog: OptimizationArtifactScopedReferenceCatalog,
    ...
) -> DefaultUclReferenceReader:
```

Reader depends only on the narrow read catalog contract.

## 8. Publisher helper typing

```python
def _publish(
    repository: OptimizationArtifactRepository,
    artifact: StoredOptimizationArtifact,
) -> None:
```

Publisher depends only on the mutable repository contract.

## 9. Default backend static conformance

Workspace B3B test:

```python
memory: _RepositoryWithScopedCatalog = InMemoryOptimizationArtifactRepository()
sqlite: _RepositoryWithScopedCatalog = SQLiteOptimizationArtifactRepository(...)
```

Pyright accepts both assignments without cast or concrete union in `_reader`.

## 10. Custom catalog static conformance

Unchanged from R1:

```python
catalog: OptimizationArtifactScopedReferenceCatalog = _CustomCatalog()
catalog: OptimizationArtifactScopedReferenceCatalog = _CustomCatalogWithListing()
```

C1: `_RecordingCatalog` remains assignable to `OptimizationArtifactScopedReferenceCatalog`.

## 11. Custom read-port static conformance

```python
port: UclReferenceReadPort = _CustomReadPort()
```

No backend classes in custom read-port proof.

## 12. Pyright result

```bash
uv run pyright \
  tests/unit/ucl/test_ucl_mp5f_b3b_workspace_scoped_reference_read.py \
  tests/unit/ucl/test_ucl_mp5f_b3b_c1_resource_scoped_reference_read.py
```

**0 errors, 0 warnings** (task gate).

## 13. Behavioral regression

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

**164 passed** (includes new static conformance test in workspace B3B module).

## 14. Production impact

**Production code changed: NO**

Only B3B test modules and this qualification document.

## 15. Final B3B verdict

B3B qualification boundary now statically proves:

- interface segregation for reader vs publisher helpers,
- default backends as dual-capability providers without reader→repository coupling,
- custom catalog and custom read port without default backend coupling.

**MP-5F-B3B — CLOSED / CERTIFIED** (pending independent audit of pushed commit).

## 16. MP-5F-B3 readiness

With B3B certified at static + behavioral gates, **MP-5F-B3** (formal UCL read boundary closure) may proceed after maintainer sign-off on this artifact and commit SHA.
