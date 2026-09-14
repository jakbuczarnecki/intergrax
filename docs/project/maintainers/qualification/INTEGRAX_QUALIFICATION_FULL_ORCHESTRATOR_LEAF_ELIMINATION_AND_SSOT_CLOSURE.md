# INTEGRAx — Full Orchestrator Leaf Elimination and SSOT Closure

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-FULL-ORCHESTRATOR-LEAF-ELIMINATION-AND-SSOT-CLOSURE` |
| Predecessor | `INTEGRAx-CANONICAL-QUALIFICATION-CATALOG-AND-FULL-ORCHESTRATOR-MIGRATION` (`6c5190758336222d52444260d0cc899ca15bbdbd`) |
| Scope | `testing_support/execution_qualification/catalog/`, legacy matrix adapters, qualification tests, this document |
| Production (`intergrax/`) | None |

## Audit Findings Addressed

| Finding | Remediation |
| --- | --- |
| MAJOR-01 — R4/Final retained orchestrator-as-leaf targets | Full recursive expansion for all canonical orchestrator paths; R2/R3 profiles compiled via expanded flat mandatory sources |
| MAJOR-02 — `mandatory_sources` imported legacy R4/Final matrices | R4/Final mandatory tuples defined in catalog; legacy matrices alias canonical sources |
| MINOR — mutable `PROFILE_BUILDERS` | `MappingProxyType` + explicit `QualificationCatalog.profile_builders` composition |

## Orchestrator Inventory

Qualification-critical modules that spawn nested pytest (inventory minimum + discovered 5F finals):

| Orchestrator test module | Nested pytest entry |
| --- | --- |
| `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` | `_run_pytest` |
| `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py` | `subprocess.run` / matrix runner |
| `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | `_run_pytest` |
| `test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py` | `_run_pytest` |
| `test_npsc5f_r2_final_journal_completeness_ordering.py` | `_run_pytest` |
| `test_npsc5f_r3_final_governed_evidence_export.py` | `_run_pytest` |

Freeze wrappers (`run_mandatory_regression_matrix`) remain legacy runners, not canonical DAG leaves.

## Canonical Orchestrator Set

Single SSOT: `CANONICAL_ORCHESTRATOR_PATHS` in `testing_support/execution_qualification/catalog/orchestrators.py` (alias `NESTED_PYTEST_ORCHESTRATOR_LEAF_PATHS` / `LEGACY_ORCHESTRATOR_EXPANSION_PATHS`).

## Expansion Model

- `orchestrator_expansion_mapping()` maps each orchestrator path → canonical `FrozenPytestSuiteSource`.
- `unique_required_leaf_targets()` expands recursively with `QualificationDependencyCycleError` on cycles.
- Termination: only non-orchestrator pytest argument vectors remain as leaves.

## R4 Expansion

`NPSC5F_R4_MANDATORY_REGRESSION_SUITES` expands R3/R2/R1 Final and NPSC-5E Final orchestrators to semantic leaves before profile compilation.

## Final Expansion

`NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES` composes R4 mandatory plus Final extras; Recovery and duplicate NPSC-5E Final entries expand through the same orchestrator map.

## SSOT Dependency Inversion

```text
Canonical Catalog (mandatory_sources)
  → legacy npsc5f_r4_regression_matrix.MANDATORY_REGRESSION_SUITES
  → legacy npsc5f_final_regression_matrix.MANDATORY_REGRESSION_SUITES
```

Catalog modules do not import legacy matrix modules or `tests.*`.

## Canonical Mandatory Sources

- `NPSC5F_R4_MANDATORY_REGRESSION_SUITES`
- `NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES`
- Existing NPSC-5E / NPSC-5F R1–R3 mandatory tuples unchanged in role; expansion registry extended for 5F finals.

## Legacy Matrix Adapters

`MANDATORY_REGRESSION_SUITES` in R4/Final matrix modules are immutable aliases of catalog tuples; `run_mandatory_regression_matrix` behavior preserved.

## Suite Identity Unification

Global `pytest_to_suite_id_registry()` enforces one `suite_id` per normalized pytest argument vector (architecture guard test).

## Catalog Immutability

- `PROFILE_BUILDERS`: `MappingProxyType`
- `QualificationCatalog`: frozen dataclass with explicit `profile_builders` mapping at composition root
- Compiled `suite_by_id`: `MappingProxyType` (unchanged)

## Anti-Regression Guards

- `test_all_canonical_profiles_are_free_of_nested_pytest_orchestrator_leaves`
- `test_no_canonical_profile_contains_any_known_orchestrator_as_leaf`
- `test_canonical_catalog_does_not_depend_on_legacy_regression_matrix_modules`
- `test_default_catalog_profile_builder_map_is_immutable`
- Parity matrix tests (recursive legacy expansion vs DAG leaf sets)
- Legacy matrix SSOT + orchestrator `_MANDATORY_SUITES` drift tests

## Parity Matrix

Recursive `unique_required_leaf_targets(mandatory)` compared to `dag_required_target_set(compiled.plan)` for:

- 5E R1/R2/R3/Final
- 5F R1/R2/R3/R4/Final

## Tests

`uv run pytest tests/unit/testing_support/execution_qualification/ -q` — required regression gate.

## Static Quality

`ruff check`, `ruff format --check`, `pyright` on catalog + legacy matrix adapters + catalog tests.

## Production Changes

None (`intergrax/` untouched).

## Findings

No remaining canonical leaf may reference a known orchestrator path. Legacy matrices are consumers only.

## Closure Decision

`FULL ORCHESTRATOR LEAF ELIMINATION + SSOT CLOSURE = PASS`

## Final Verdict

Canonical qualification catalog is the sole maintainer of R4/Final mandatory composition; nested orchestrator leaves are eliminated from all compiled profiles; dependency direction and guards are aligned for semantic parity certification.
