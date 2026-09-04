# DIAG-FUNCTIONAL-H1 TEST-SUITE HEALTH QUALIFICATION

## Canonical command

From repository root:

```bash
uv run python tests/system/functional_diagnostics_h1/runner.py
```

Unit/meta gates (no full qualification):

```bash
uv run pytest tests/unit/system/functional_diagnostics_h1 tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py -q
```

## H1 semantics

H1 measures **diagnostic test-suite health**, not live requalification of Q1–Q5/D1/S1/R1 historical proofs.

- `CORE_TEST_HEALTH` = deterministic inventory, collection, invariant ownership, repeatability, runner integrity
- `REAL_SERVICE_QUALIFICATION_AVAILABILITY` = preflight classification only (READY/BLOCKED explicit)
- External service absence does **not** override a core FAIL, but is reported separately as NOT REVALIDATED / BLOCKED

## First canonical run result (2026-09-04)

| Field | Value |
| --- | --- |
| Verdict | **FAILED** |
| Start HEAD | `33ae7d2d1e878d97758a5764ba975f9702c7c9cf` |
| Final HEAD | `33ae7d2d1e878d97758a5764ba975f9702c7c9cf` |
| Machine artifact | `.tmp/session/diag-functional-h1/qualification-report.json` |
| Inventory artifact | `.tmp/session/diag-functional-h1/test-inventory.json` |

### Gate summary

| Gate | Result |
| --- | --- |
| H1-A collection | PASS (442 tests collected, 0 collection errors, 92 inventory files) |
| H1-B core health | **FAILED** (359 passed, 5 failed) |
| H1-C repeatability | PASS (3×111 passed, stable) |
| H1-D invariant coverage | PASS (23/23 normative owners) |
| H1-E skip/xfail honesty | PASS (11 justified markers) |
| H1-F external dependency | PASS (explicit READY/BLOCKED) |
| H1-G runner integrity | PASS |
| H1-H stale/dead | PASS |
| H1-I supersession (R1→R1-R3) | PASS |
| H1-J report integrity | PASS |

### Blocking failures (H1-R1 scope)

Five architecture gate failures in core deterministic suite:

1. `tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py::test_df4_scenario_task_preserves_run_and_uses_terminal_diagnostics`
2. `tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py::test_df4_hosted_application_uses_injected_orchestrator_subject_scope`
3. `tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py::test_df4_scenario_runtime_has_no_separate_diagnostic_engine`
4. `tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py::test_df5_case_d_production_scenario_without_diagnostics_fails`
5. `tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py::test_df5_case_d_production_scenario_with_diagnostics_attaches`

Observed failure modes:

- `ApplicationPackageClosureError` — host tool registry missing required catalog tools during scenario/host composition
- `ProblemLifecycleEngine.__init__()` missing `occurrence_persistence` in hosted diagnostic integration helper
- `test_df4_scenario_runtime_has_no_separate_diagnostic_engine` — `handle_task` no longer present in `scenario_runtime_baseline` source contract under test

Local integration diagnostics (Phase E) blocked by host composition prerequisites (`SkillToolRequirementError` / package closure) — classified as environment/composition BLOCKED, not PASS.

### External preflight (not re-run)

| Family | Status |
| --- | --- |
| Q1 | BLOCKED_SERVICE_UNAVAILABLE |
| Q2 | BLOCKED_SERVICE_UNAVAILABLE |
| Q3 | BLOCKED_MISSING_CREDENTIAL |
| Q4 | READY |
| Q5 | READY |
| D1/S1/R1/R1-R1/R1-R2/R1-R3 | READY (Mongo reachable) |

Historical qualifications remain valid; live revalidation was **not executed** in this H1 run.

## Infrastructure delivered

```text
tests/system/functional_diagnostics_h1/
  models.py          # typed health/report models
  inventory.py       # DiagnosticTestDescriptor inventory + invariant matrix
  gates.py           # H1-A..J gate implementations
  preflight.py       # external dependency classification
  composition.py     # qualification family enumeration
  reporting.py       # machine + human report serialization
  runner.py          # canonical orchestrator (subprocess pytest)
  test_h1_architecture_gates.py
tests/unit/system/functional_diagnostics_h1/
  test_h1_runner.py  # synthetic FAIL/BLOCKED classification proofs
```

## Remediation path

Per H1 protocol: **do not re-run under the same label after material gate failure**. Open **H1-R1** to repair architecture gate drift (DF4/DF5 + host composition prerequisites), then re-qualify.

## H1-R1 remediation (2026-09-04)

| Field | Value |
| --- | --- |
| Qualification id | `DIAG-FUNCTIONAL-H1-R1` |
| Verdict | **PASS** |
| Original H1 verdict | **FAILED** (preserved above) |
| Qualified SHA | see `DIAG_FUNCTIONAL_H1_R1_TEST_SUITE_HEALTH_QUALIFICATION.md` |
| Machine artifact | `.tmp/session/diag-functional-h1-r1/qualification-report.json` |

H1-R1 repaired five DF4/DF5 architecture blockers (package closure, lifecycle contract drift, stale source gate) without weakening diagnostic invariants. First canonical H1 **FAILED** result remains immutable audit evidence.
