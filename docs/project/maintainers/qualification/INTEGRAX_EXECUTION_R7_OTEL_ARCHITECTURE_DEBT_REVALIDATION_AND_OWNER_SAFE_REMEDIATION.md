# INTEGRAx Execution R7 — OTel Architecture Debt Revalidation and Owner-Safe Remediation

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-R7-OTEL-ARCHITECTURE-DEBT-REVALIDATION-AND-OWNER-SAFE-REMEDIATION` |
| Group | OTEL (ARCH-F19) |
| Qualification date | 2026-09-16 |
| Branch | `development` |
| Mode | CURRENT-STATE REVALIDATION + QUALIFICATION HARNESS REMEDIATION |

## Scope

Revalidate ARCH-F19 (`test_direct_opentelemetry_imports_are_allowlisted`) on committed HEAD. Remediate only stale HARDEN-3E allowlist drift for canonical OTLP optional-dependency boundary (`otlp_dependency.py`). **No** edits to `intergrax/runtime/observability/**` production modules, execution chain, governance, diagnostics production paths, or observability contracts.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| Baseline SHA (start) | `51df4755a0ae791c5ccc2a88b19780aa56233688` |
| `origin/development` | `51df4755a0ae791c5ccc2a88b19780aa56233688` |
| Dirty tracked (unrelated WIP) | `intergrax/contracts/execution_continuation.py`, `intergrax/runtime/decision_flow.py`, `testing_support/mp4r7_*`, several `tests/unit/*` |
| Untracked (unrelated) | `intergrax/contracts/decision/integration/execution_continuation.py`, `intergrax/tools/*`, `tests/unit/memory/*` |
| Stash | `stash@{0..2}` (not applied) |

F19 revalidation executed on main tree; WIP does not touch OTel scan targets or R7 changed files.

## Baseline SHA

`51df4755a0ae791c5ccc2a88b19780aa56233688` — platform HEAD at R7 start.

## Protected Surface Inventory

| Surface / subsystem | Dirty? | Parallel work? | R7 ownership? | Editable? |
| --- | ---: | ---: | ---: | ---: |
| `intergrax/runtime/observability/**` | no | yes | observability | **PROTECTED** |
| `intergrax/runtime/diagnostics/**` | no | yes | diagnostics | **PROTECTED** |
| `intergrax/runtime/events/**` | no | yes | events | **PROTECTED** |
| `intergrax/runtime/execution/**` | no | yes | execution | **PROTECTED** |
| `intergrax/runtime/nexus/**` | no | yes | nexus | **PROTECTED** |
| `intergrax/runtime/task/**` | no | yes | task | **PROTECTED** |
| `intergrax/applications/**` | no | yes | applications | **PROTECTED** |
| `intergrax/contracts/**` | **yes** (continuation WIP) | yes | contracts | **PROTECTED** |
| `testing_support/obs_*`, `diagnostic*` | partial (mp4r7) | yes | mixed | **PROTECTED** |
| HARDEN-3E OTel import gate | R7 | no | qualification | **yes** |

## Parallel Workstream Risk Assessment

Decision-flow / mp4r7 / memory ENT-14 WIP is unrelated to F19. R7 touches only `test_harden_3e_otel_import_gate.py` and this qualification artifact.

## Historical ARCH-F19

Documented in `INTEGRAX_EXECUTION_FULL_ARCHITECTURE_SUITE_FAILURE_DIAGNOSTICS_AND_BLOCKER_CLASSIFICATION.md`: direct `opentelemetry` import scan reported three new `opentelemetry.sdk._logs` / OTLP log exporter import sites in `otlp_dependency.py` outside the frozen allowlist (classification **E**).

## Exact Historical Node Mapping

| ID | Group | Exact test node | Historical symptom |
| --- | --- | --- | --- |
| ARCH-F19 | OTEL | `tests/unit/runtime/architecture/test_harden_3e_otel_import_gate.py::test_direct_opentelemetry_imports_are_allowlisted` | 3 new `opentelemetry.sdk._logs` (+ OTLP log exporter) import sites in `otlp_dependency.py` not allowlisted |

## OTel Test Inventory

| Test / module | Purpose | Owner |
| --- | --- | --- |
| `test_harden_3e_otel_import_gate.py` | HARDEN-3E direct OTel import allowlist gate (F19) | Execution qualification |
| `test_harden_3f_qualification_matrix.py` | HARDEN-3 matrix registry (OTel row) | Execution qualification |
| `test_rag_otel_spans.py` | RAG tracking span adapter | RAG / derived observability |
| `test_context_otel_spans.py` | Context tracking span adapter | Context / derived observability |
| `test_diag_final_external_otel_e2e.py` | Diagnostic E2E OTel spine (integration) | Diagnostics qualification |
| `tests/unit/runtime/observability/**` | Observability subsystem unit suite | Observability |
| OBS-DIAG `obs_diag_conformance` marker suite | Port + architecture conformance | OBS-DIAG qualification |

## Current Revalidation

```bash
uv run pytest tests/unit/runtime/architecture/test_harden_3e_otel_import_gate.py::test_direct_opentelemetry_imports_are_allowlisted -q
```

| Test | Run 1 (pre-fix) | Run 2 (pre-fix) | Run 1 (post-fix) | Run 2 (post-fix) |
| --- | --- | --- | --- | --- |
| F19 | FAIL | (deterministic; not re-run) | PASS | PASS |

Pre-fix assertion: `violations == []` — left contained `intergrax/runtime/observability/exporters/otlp/otlp_dependency.py:29` (and two sibling OTLP log exporter probes).

## Actual Call Chain

```text
pytest (F19)
  → _production_python_files() scan intergrax/, agents/, applications/
  → ast.parse per file
  → _collect_direct_otel_imports
  → rel not in _DIRECT_OTEL_ALLOWED_FILES → violation
```

Production OTLP boundary (observed, not modified):

```text
W5-H1 optional profile check
  → otlp_dependency.require_otlp_observability_dependency_profile()
  → lazy import opentelemetry.sdk._logs + OTLP log exporters (ImportError → ConfigurationError)
```

## Failure Inventory

| ID | Symptom | Layer | Owner |
| --- | --- | --- | --- |
| ARCH-F19 | Allowlist missing `otlp_dependency.py` | qualification gate | HARDEN-3E |

## Canonical Owner Mapping

| Mechanism | Canonical owner |
| --- | --- |
| Span creation (runtime export path) | Observability exporters / adapters |
| Trace context propagation | Runtime events + W3C headers on envelopes |
| Exporter selection | Observability composition + `ExporterKind` contracts |
| Telemetry configuration | Application environment profile + observability wiring |
| Execution→telemetry integration | Observability export boundary (non-authoritative) |
| Direct OTel SDK import policy | HARDEN-3E architecture gate (qualification) |

## Contract Mapping

| Contract | Role |
| --- | --- |
| `intergrax.contracts.observability_export` | Export kinds, configuration errors |
| `OtlpTransportPort` / OTLP adapter package | Vendor SDK behind observability exporter layer |
| HARDEN-3E allowlist | Documents sole approved direct-import sites |

## Layer Ownership Mapping

| Layer | OTel touch | R7 edit |
| --- | --- | --- |
| `intergrax/runtime/execution/**` | none (gate enforces) | none |
| `intergrax/runtime/observability/exporters/otlp/**` | adapter + dependency probe | none (read-only) |
| `intergrax/rag|context/tracking/*_spans.py` | allowlisted adapters | none |
| `tests/.../test_harden_3e_otel_import_gate.py` | gate | allowlist update |

## Trace Context Model

`trace_id` / `traceparent` on runtime events are evidence correlation fields; distinct from `run_id`, `task_id`, `execution_id` (execution identity authority).

## Execution Identity Correlation Assessment

Telemetry may correlate execution IDs when present on events; F19 does not exercise identity minting. No R7 change to identity semantics.

## OTel Provider / Exporter Assessment

OTLP SDK imports confined to observability exporter adapter package and allowlisted tracking modules. `otlp_dependency.py` performs optional-package presence checks only (fail-closed `ConfigurationError`), not export at execution hot path.

## Optional Dependency Assessment

OpenTelemetry packages are profile-gated (`observability-otlp`); lazy import in dependency boundary is intentional (W5-H1). Gate must allowlist the boundary module, not require installing OTel in default test env.

## F19 Analysis

Failure is **not** missing telemetry at runtime; it is **allowlist drift** after adding W5-H1 log-exporter dependency probes alongside existing `otlp_transport.py` span transport site.

## Root Cause Classification

**E** (FROZEN / REFERENCE DRIFT) with secondary alignment to **C** (STALE TEST EXPECTATION) — allowlist not updated when canonical `otlp_dependency.py` was added.

## Dirty vs Clean Assessment

| Target | Dirty tree | Clean worktree | Classification |
| --- | --- | --- | --- |
| F19 | FAIL pre-fix | not required | Harness drift on committed observability layout; unrelated WIP does not affect AST scan targets |

## Editable vs Protected Surfaces

| Surface | Protected? | Touched by R7? |
| --- | ---: | ---: |
| `intergrax/runtime/observability/**` | yes | no |
| `intergrax/runtime/execution/**` | yes | no |
| HARDEN-3E gate test | no | yes |
| R7 qualification doc | no | yes |

## Selected Remediation

Add `intergrax/runtime/observability/exporters/otlp/otlp_dependency.py` to `_DIRECT_OTEL_ALLOWED_FILES` (same adapter layer as `otlp_transport.py`).

## Deferred Cross-Layer Items

ARCH-F20+ (HARDENING-3 contracts, harness L3 paths, maturity, NPSC pins, UE drift items) remain outside R7 scope.

## Contracts / Ports

No contract changes. Reused existing HARDEN-3E allowlist mechanism.

## Pluginability Assessment

Unchanged — OTLP remains optional profile behind adapter boundary.

## Configuration / Composition Assessment

Unchanged — explicit `require_otlp_observability_dependency_profile` and composition-root observability wiring.

## Global State Assessment

No global tracer/provider mutations introduced.

## Persistence Boundary Assessment

N/A for F19.

## Events Boundary Assessment

N/A for F19 (gate is static import scan).

## Diagnostics Boundary Assessment

Diagnostics may consume OTel evidence in separate suites; not modified.

## Evidence vs Control Assessment

F19 validates **import surface control** for qualification, not execution ALLOW/DENY. Telemetry remains non-authoritative.

## Failure Semantics Assessment

`otlp_dependency` raises `ConfigurationError` when profile packages absent — observability configuration fail-closed, not execution denial.

## Layer Boundary Assessment

No new cross-tier imports. Gate scan unchanged.

## Frozen Invariant Assessment

`Evidence != Control` preserved. Execution chain untouched.

## Execution Engine Impact

**None** — test-only allowlist alignment.

## Architecture Reopen Assessment

**Not required** — no ownership or public execution contract change.

## Targeted Results

F19: **PASS ×2** post-fix.

## R1 Regression

F01–F04, F06 (`test_audit_ideal_depth_gate` nodes): **PASS** (`.tmp/session/R7-OTEL/regression-main.log`).

## R3 Regression

F13, F14 (`test_diag_foundation_4_entrypoint_consistency` nodes): **PASS** (same log).

## R4 Regression

F15, F16: **PASS** (same log).

## R6 Regression

F18: **PASS** (same log).

## R14 Regression

F34–F37 (UE gate nodes): **PASS** (same log).

## R9 Regression

`test_mandatory_frozen_suite_passes[NPSC-5E Final]`: **PASS** (`.tmp/session/R7-OTEL/regression-npsc-frozen.log`).

## R10 Regression

`test_mandatory_frozen_suite_passes[NPSC-5D Final]`: **PASS** (same log).

## R11 Regression

`Runtime events suites`, `Runtime observability suites`, `DG_001` mandatory frozen rows: **PASS** (same log).

## OBS-DIAG Regression

`test_obs_diag_conformance_architecture.py`, `test_obs_diag_conformance_qualification.py`, `test_obs_diag_port_1_gates.py`: **PASS** (regression-main log).

## OTel Regression Bundle

Full `test_harden_3e_otel_import_gate.py`, `test_rag_otel_spans.py`, `test_context_otel_spans.py`: **PASS** (regression-main log).

## EE Architecture Gates

`tests/unit/runtime/architecture/` `-k test_ee_final_arch`: **29 passed** (`regression-ee-final-arch.log`).

## U5

`test_platform_execution_unification_u5_final_zero_bypass.py`: **PASS** (regression-main log).

## UE-10R4.1

`test_ue_10r41_execution_import_hygiene_gate.py`: **PASS** (regression-main log).

## F-01

`test_ee_b2_worker_fault_injection.py` (F-01 fault matrix module): **PASS** (regression-main log).

## R5 Boundary

`test_intergrax_no_applications_import_gate`: **PASS** (regression-main log).

## Qualification Regression

`tests/unit/testing_support/execution_qualification/`: **PASS** (7 skips live perf env only).

## Runtime Unit Regression

**N/A** — no production runtime edits.

## Static Quality

| Check | Result |
| --- | --- |
| `ruff check` (`test_harden_3e_otel_import_gate.py`) | PASS |
| `ruff format` (`test_harden_3e_otel_import_gate.py`) | PASS (formatted) |
| `git diff --check` (R7 file) | PASS |

## Cold Imports

**N/A** — no production module changes.

## Changed Files

| File | Layer | Reason | Contract changed? | Boundary-safe? |
| --- | --- | --- | ---: | ---: |
| `tests/unit/runtime/architecture/test_harden_3e_otel_import_gate.py` | qualification | Allowlist `otlp_dependency.py` | no | yes |
| `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_R7_OTEL_ARCHITECTURE_DEBT_REVALIDATION_AND_OWNER_SAFE_REMEDIATION.md` | docs | R7 artifact | no | yes |

## Untouched Protected Files

All observability/diagnostics/execution production paths and parallel WIP listed in Repository State.

## Remaining Debt

ARCH-F20+ unchanged. F19 closed for R7.

## Decision

Close F19 via HARDEN-3E allowlist alignment; preserve observability adapter ownership and execution freeze.

## Commit SHA

`d2b4a2ba0423e2939db33b89baf36dde64a4c39f`

## Final Verdict

```text
R7 OTEL = PASS — QUALIFICATION DEBT REMEDIATED, OBSERVABILITY ARCHITECTURE PRESERVED
```

## Required F19 table

| ID | Exact test | Current result | Root cause | Canonical owner | R7 action |
| --- | --- | --- | --- | --- | --- |
| ARCH-F19 | `test_direct_opentelemetry_imports_are_allowlisted` | PASS | E — allowlist drift for `otlp_dependency.py` | HARDEN-3E qualification | Add file to allowlist |

## Required telemetry contract table

| Mechanism | Platform contract | Canonical owner | Default implementation | Replaceable? |
| --- | --- | --- | --- | ---: |
| Export envelope | `observability_export` | Observability | OTLP / journal adapters | yes |
| OTLP transport | `OtlpTransportPort` | Observability exporters | `otlp_transport.py` | yes |
| Optional OTLP deps | W5-H1 profile gate | Observability exporters | `otlp_dependency.py` | yes (profile) |
| Direct SDK import policy | HARDEN-3E gate | Qualification | allowlist in test | n/a |

## Required trace identity table

| Identifier | Owner | Minted by | Used by telemetry as |
| --- | --- | --- | --- |
| `trace_id` / traceparent | Event envelope / propagation | upstream context | correlation attribute |
| `run_id` | Execution identity | authority | correlate only |
| `task_id` | Execution identity | authority | correlate only |
| `execution_id` | Execution identity | authority | correlate only |

## Required provider table

| Component | Provider-neutral? | Concrete dependency | Correct boundary? |
| --- | ---: | --- | ---: |
| Core execution | yes | none OTel | yes |
| OTLP adapter | no (explicit OTLP) | opentelemetry SDK (optional) | yes — exporter layer |
| RAG/context spans | no (OTel adapter) | opentelemetry | yes — allowlisted tracking |

## Required failure semantics table

| Telemetry failure | Expected effect on execution | Contract evidence |
| --- | --- | --- |
| Missing OTLP profile packages | Export/config error; no execution deny | `ConfigurationError` in `otlp_dependency` |
| Exporter failure | Best-effort / bounded per observability policy | observability subsystem (unchanged) |

## Required boundary table

| Dependency | Allowed? | Existing/new | Verdict |
| --- | ---: | --- | --- |
| qualification → AST read `intergrax/**` | yes | existing | OK |
| qualification → mutate observability impl | no | — | not introduced |

## Required protected table

See Editable vs Protected Surfaces.

## Required changed-files table

See Changed Files.
