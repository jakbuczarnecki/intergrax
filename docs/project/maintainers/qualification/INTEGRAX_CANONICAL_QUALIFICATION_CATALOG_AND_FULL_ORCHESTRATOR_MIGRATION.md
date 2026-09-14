# INTEGRAx-CANONICAL-QUALIFICATION-CATALOG-AND-FULL-ORCHESTRATOR-MIGRATION

## Metadata

| Field | Value |
| ----- | ----- |
| Task ID | `INTEGRAx-CANONICAL-QUALIFICATION-CATALOG-AND-FULL-ORCHESTRATOR-MIGRATION` |
| Base HEAD | `bb3533617f3f078727f891605b482b27bc3f156a` |
| Branch | `development` |

## Scope

Establish `testing_support/execution_qualification/catalog/` as the single source of truth for qualification suite definitions, gate graphs, and profile roots. Invert dependency so canonical runtime no longer imports `tests.unit.runtime.architecture.*` for SSOT. Migrate NPSC-5E/5F qualification-critical orchestrator flows to compiled canonical DAG profiles with target-set parity proofs.

Out of scope: production `intergrax/` runtime changes, cross-run cache, wall-time certification, EE-B2/B3/B4.

## Legacy Orchestrator Inventory

| Flow | Module / matrix | Classification |
| ---- | ---------------- | -------------- |
| NPSC-5E R1 Final | `test_npsc5e_r1_final_retry_attempt_qualification.py` | `STATIC_ONLY` (in-process gates; catalog leaf = whole file) |
| NPSC-5E R2 Final | `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` | `READY_FOR_CATALOG` → `npsc5e-r2-final` |
| NPSC-5E R3 Final | `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py` | `READY_FOR_CATALOG` → `npsc5e-r3-final` |
| NPSC-5E Final | `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | `READY_FOR_CATALOG` → `npsc5e-final` |
| NPSC-5F R1 Final | `test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py` | `REPRESENTATIVE_ALREADY_MIGRATED` → catalog-backed `npsc5f-r1-final` |
| NPSC-5F R2 Final | `test_npsc5f_r2_final_journal_completeness_ordering.py` | `READY_FOR_CATALOG` → `npsc5f-r2-final` |
| NPSC-5F R3 Final | `test_npsc5f_r3_final_governed_evidence_export.py` | `READY_FOR_CATALOG` → `npsc5f-r3-final` |
| NPSC-5F R4 Final | `npsc5f_r4_regression_matrix.py` | `READY_FOR_CATALOG` → `npsc5f-r4-final` |
| NPSC-5F Final | `npsc5f_final_regression_matrix.py` | `READY_FOR_CATALOG` → `npsc5f-final` |

## Canonical Catalog Architecture

```text
testing_support/execution_qualification/catalog/
  contracts.py          # CompiledCatalogProfile
  mandatory_sources.py  # frozen (label, pytest targets) SSOT
  labels.py             # semantic suite_id maps
  suite_registry.py     # pytest args → suite_id (fail-closed)
  expansion.py          # orchestrator expansion (legacy parity paths)
  profile_builders.py   # profile → QualificationGraphDefinition
  catalog.py            # QualificationCatalog.compile_profile
  composition.py        # build_default_qualification_catalog()
```

## Dependency Inversion

| Before | After |
| ------ | ----- |
| `npsc5f_r1_qualification_profile` imported `tests.unit.*` `_MANDATORY_SUITES` | Profile builders read `mandatory_sources.py` only |
| Label maps lived only in test helper modules | `catalog/labels.py` is SSOT; test modules re-export where needed |

Legacy test modules retain `_MANDATORY_SUITES` for reference; `legacy_reference_mandatory_source_from_tests()` supports drift detection.

## Suite Catalog

Shared semantic IDs: `dg001-lineage`, `npsc5d-final`, `runtime-events`, `runtime-observability`, plus flow-scoped `npsc5e-r3.mandatory.*` / `npsc5f-r*.*` entries.

## Gate Catalog

Aggregate gates only consume suite receipts via existing `aggregate.py` (no subprocess).

## Profiles

| profile_id | root_gate_id |
| ---------- | ------------ |
| `npsc5e-r1-final` | `npsc5e-r1.final` |
| `npsc5e-r2-final` | `npsc5e-r2.final` |
| `npsc5e-r3-final` | `npsc5e-r3.final` |
| `npsc5e-final` | `npsc5e.final` |
| `npsc5f-r1-final` | `npsc5f-r1.final` |
| `npsc5f-r2-final` | `npsc5f-r2.final` |
| `npsc5f-r3-final` | `npsc5f-r3.final` |
| `npsc5f-r4-final` | `npsc5f-r4.final` |
| `npsc5f-final` | `npsc5f.final` |

## Composition

`build_default_qualification_catalog()` returns immutable `QualificationCatalog` with explicit profile list (no dynamic registry).

## Migrated Flows

All nine profiles above compile via `QualificationCatalog.compile_profile` and pass target-set parity tests in `test_catalog_parity_matrix.py`.

## Blocked Flows

None for inventory targets. NPSC-5E R1 remains a single-file leaf profile (no subprocess matrix).

## Target-Set Parity Matrix

| Legacy Flow | Canonical Profile | Parity test |
| ----------- | ----------------- | ----------- |
| NPSC-5F R1 Final | `npsc5f-r1-final` | `test_parity_npsc5f_r1_final` |
| NPSC-5E R3 Final | `npsc5e-r3-final` | `test_parity_npsc5e_r3_final` |
| NPSC-5E R2 Final | `npsc5e-r2-final` | `test_parity_npsc5e_r2_final` |
| NPSC-5E R1 Final | `npsc5e-r1-final` | `test_parity_npsc5e_r1_final` |
| NPSC-5E Final | `npsc5e-final` | `test_parity_npsc5e_final` |
| NPSC-5F R2 Final | `npsc5f-r2-final` | `test_parity_npsc5f_r2_final` |
| NPSC-5F R3 Final | `npsc5f-r3-final` | `test_parity_npsc5f_r3_final` |
| NPSC-5F R4 Final | `npsc5f-r4-final` | `test_parity_npsc5f_r4_final` |
| NPSC-5F Final matrix | `npsc5f-final` | `test_parity_npsc5f_final` |

## Deduplication Matrix

Representative `npsc5f-r1-final`: shared `dg001-lineage` physical execution count = 1 (`test_each_leaf_executes_at_most_once_per_plan_run`).

## Nested-Orchestrator Elimination

Catalog leaves exclude the three legacy expansion orchestrator paths (`LEGACY_ORCHESTRATOR_EXPANSION_PATHS`). Guard: `test_canonical_qualification_catalog_has_no_nested_pytest_orchestrator_leaves`.

## Legacy Compatibility Layer

`testing_support/npsc5f_r1_legacy_targets.py` uses catalog mandatory sources; optional `legacy_reference_mandatory_source_from_tests()` for parity/reference path.

## Immutability

`CompiledCatalogProfile.suite_by_id` is `MappingProxyType`; catalog modules use frozen dataclasses and explicit tuples.

## Generic Core Reuse

Unchanged: `compile_qualification_execution_plan`, `QualificationPlanRunner`, coordinator, subprocess executor.

## Tests

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
```

## Static Quality

`ruff check`, `ruff format --check`, `pyright` on catalog + touched wrappers.

## Production Changes

None (`intergrax/` untouched).

## Findings

- Orchestrator file targets in 5F R2/R3 use profile-local suite IDs while sharing pytest args with other profiles; direct-branch suite construction uses label maps to avoid registry collisions.
- NPSC-5F Final matrix remains logically multi-request but canonical plan deduplicates shared suite IDs.

## Migration Status

All inventory flows have canonical profiles; legacy modules remain as reference for global semantic parity certification (next task).

## Final Verdict

**CANONICAL QUALIFICATION CATALOG + FULL ORCHESTRATOR MIGRATION = PASS WITH OBSERVATIONS**

Observation: legacy `_MANDATORY_SUITES` tuples in test modules not yet generated from catalog (reference-only); optional follow-up is codegen or import-from-catalog in test modules.
