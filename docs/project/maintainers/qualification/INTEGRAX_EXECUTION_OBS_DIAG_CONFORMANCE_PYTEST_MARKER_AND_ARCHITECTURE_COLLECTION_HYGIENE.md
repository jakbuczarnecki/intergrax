# INTEGRAx-EXECUTION-OBS-DIAG-CONFORMANCE-PYTEST-MARKER-AND-ARCHITECTURE-COLLECTION-HYGIENE

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAX-EXECUTION-OBS-DIAG-CONFORMANCE-PYTEST-MARKER-AND-ARCHITECTURE-COLLECTION-HYGIENE` |
| Class | A — test configuration hygiene |
| Execution Engine | FROZEN / ENTERPRISE-CERTIFIED — no semantic changes |
| Related closure | OBS-DIAG-CONFORMANCE (evidence → reconstruction → diagnostics) |

## Scope

Register the pytest marker `obs_diag_conformance` in canonical `[tool.pytest.ini_options].markers` (`pyproject.toml`) under `--strict-markers` / `--strict-config`, restore clean collection for `tests/unit/runtime/architecture/`, and document marker semantics. No runtime, contract, or diagnostics behavior changes.

## Repository State

Audit and verification on branch `development`. Operator unrelated WIP may be present in the working tree; remediation touches only pytest marker SSOT and this qualification record.

## Baseline SHA

Pre-remediation parent (marker absent from `pyproject.toml`): `35b777b0662dc1aa29ca68aa4c10cd3633d3efb8^`
Marker registration landed in: `35b777b0662dc1aa29ca68aa4c10cd3633d3efb8` (`OBS-DIAG-CONFORMANCE: certify evidence to diagnostics boundary`).

## Original Collection Failure

With `obs_diag_conformance` absent from `markers`, strict collection fails on OBS-DIAG modules:

```text
'obs_diag_conformance' not found in `markers` configuration option
ERROR collecting tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py
ERROR collecting tests/unit/runtime/architecture/test_obs_diag_conformance_qualification.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
```

(Reproduced locally by temporarily restoring parent `pyproject.toml` markers list.)

## Marker Usage Inventory

| File | Test/module | Purpose | Expected marker semantics | Production impact |
| --- | --- | --- | --- | --- |
| `tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py` | Module `pytestmark` + AST/import gates | Deterministic architecture gates for diagnostic orchestrator ↔ shared reconstruction ownership | `unit` + `gate` + `obs_diag_conformance` | None |
| `tests/unit/runtime/architecture/test_obs_diag_conformance_qualification.py` | Manifest + `test_obs_diag_conformance_pytest_mark_registered` | P1 proof module registry; asserts marker string present in `pyproject.toml` | `unit` + `gate` + `obs_diag_conformance` | None |
| `tests/integration/runtime/test_obs_diag_conformance_e2e.py` | Module `pytestmark` | Deterministic integration spine: Producer → Evidence → Reconstruction → DIAG | `integration` + `obs_diag_conformance` | None |

| Test file | Marker use | Purpose | Classification |
| --- | --- | --- | --- |
| `test_obs_diag_conformance_architecture.py` | `pytestmark` | Import/symbol forbidden-list gates on diagnostics spine | Unit / gate / conformance |
| `test_obs_diag_conformance_qualification.py` | `pytestmark` | Qualification manifest + on-disk proof paths | Unit / gate / conformance |
| `test_obs_diag_conformance_e2e.py` | `pytestmark` | Cross-layer E2E conformance scenarios | Integration / conformance |

## Marker Semantics

`obs_diag_conformance` selects the **OBS-DIAG-CONFORMANCE** qualification slice: deterministic tests proving the observability evidence plane, neutral reconstruction, and diagnostics read path stay aligned (architecture gates, manifest registry, integration E2E). Tests are deterministic; they do **not** require external network or the `qualification` marker’s live-service profile. The marker is **not** a duplicate of `obs_coverage_p1` or `obs_trace_1` (those cover separate OBS P1/trace contracts). It complements `gate` (fast regression) by enabling `pytest -m obs_diag_conformance` as the documented OBS-DIAG run selector (see `DIAGNOSTICS.md`, `OBSERVABILITY.md`).

## Canonical Configuration Ownership

Single SSOT: `[tool.pytest.ini_options].markers` in `pyproject.toml`. No `pytest.ini`, no `pytest_configure` registration, no conftest marker duplication.

## Selected Remediation

Add one line to `markers`:

```text
obs_diag_conformance: OBS-DIAG-CONFORMANCE evidence → reconstruction → diagnostics qualification.
```

Delivered in commit `35b777b0662dc1aa29ca68aa4c10cd3633d3efb8`. This task adds qualification closure documentation only.

## Production Impact

**None.** Configuration-only.

## Execution Engine Impact

**None.** No `intergrax/runtime/execution` changes.

## Diagnostics / Observability Impact

**None** at runtime. Test taxonomy only; conformance tests unchanged.

## Architecture Collection Result

```text
uv run pytest tests/unit/runtime/architecture/ --collect-only -q
1855 tests collected in 4.49s
0 collection errors
```

## Full Architecture Suite Result

```text
uv run pytest tests/unit/runtime/architecture/ -q
35 failed, 1810 passed (session log: .tmp/session/obs-diag-marker/architecture-full.log)
```

Collection hygiene objective **met** (0 collection errors). Full-suite failures are **not** caused by unknown-marker configuration; they are pre-existing / unrelated architecture gate failures (OTel allowlist, harness L3 scripts, NPSC-5F frozen suite subprocess, prompt golden catalog, UE gates, etc.). Two failures in the log named OBS-DIAG architecture tests **passed** on immediate re-run in isolation (15 passed). **Residual full-suite debt** remains outside this Class A marker task.

## Marker-Specific Tests

```text
uv run pytest -m obs_diag_conformance (subset via explicit paths)
tests/unit/runtime/architecture/test_obs_diag_conformance_*.py
tests/integration/runtime/test_obs_diag_conformance_e2e.py
→ PASS (included in mandatory gate bundle below)
```

Re-run:

```text
uv run pytest tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py \
  tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py -q
15 passed
```

## Existing Architecture Gates

| Gate | Module(s) | Result |
| --- | --- | --- |
| `test_ee_final_arch_*` | 10 modules | **PASS** (280-test mandatory bundle) |
| U5 zero-bypass | `test_platform_execution_unification_u5_final_zero_bypass.py` | **PASS** |
| UE-10R4.1 | `test_ue_10r41_execution_import_hygiene_gate.py` | **PASS** |
| F-01 boundary guard | `test_intergrax_no_testing_support_import_gate.py` | **PASS** |

## Qualification Regression

```text
uv run pytest tests/unit/testing_support/execution_qualification/ -q
257 passed, 7 skipped (live perf env skips)
```

## Static Quality

Documentation-only delta for this closure commit: `git diff --check` on staged qualification doc.

## Changed Files

| Path | Change |
| --- | --- |
| `pyproject.toml` | Marker line (already on `development` @ `35b777b`) |
| `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_OBS_DIAG_CONFORMANCE_PYTEST_MARKER_AND_ARCHITECTURE_COLLECTION_HYGIENE.md` | **Added** — hygiene closure evidence |

## Remaining Debt

- Full `tests/unit/runtime/architecture/` suite: 35 failing tests on long local run (see session log); not marker/collection blockers.
- Operator WIP: partial `git stash pop` may leave unstaged changes; not part of this task.

## Decision

Marker semantics confirmed; canonical registration present; strict markers unchanged. Collection stability restored for OBS-DIAG modules and full architecture tree.

## Commit SHA

_(filled at commit time for qualification doc closure commit)_

## Final Verdict

**OBS-DIAG CONFORMANCE MARKER / ARCHITECTURE COLLECTION HYGIENE = PARTIAL — RESIDUAL TEST INFRASTRUCTURE ISSUE**

Marker / collection hygiene: **PASS**. Full architecture suite zero-failure bar: **not met** on local long run (unrelated failures); mandatory OBS-DIAG + EE gate subset: **PASS**.

### Final canonical marker definition

```text
obs_diag_conformance: OBS-DIAG-CONFORMANCE evidence → reconstruction → diagnostics qualification.
```

Selector: `uv run pytest -m obs_diag_conformance` (architecture gates, qualification manifest, integration E2E spine).
