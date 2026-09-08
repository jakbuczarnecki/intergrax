# NPSC-4.2-F — Residual Compatibility Surface Final Certification

**Verdict:** FAIL  
**Date:** 2026-09-08  
**Branch:** `development`

## Certification identity

| Field | Value |
|---|---|
| CERTIFIED_CODE_HEAD | `42f10358a12602d2727ba83cb5bcf38a09aa8c87` |
| REMOTE_AT_START | `42f10358a12602d2727ba83cb5bcf38a09aa8c87` |
| REMOTE_AT_END | `42f10358a12602d2727ba83cb5bcf38a09aa8c87` |
| CERTIFICATION_COMMIT | NONE (FAIL — not committed) |

## Preflight

- branch: `development`
- HEAD == origin/development: YES
- worktree clean: YES

## Intervening commits since R1 (`edc69d530bc277ddedb8e105f7bbc7dc3de36002`)

**INTERVENING_SENSITIVE_CHANGES:** NONE

Changed files (all VPI / embedding / unrelated):

- `intergrax/rag/embedding/providers/hf_embedding_provider.py`
- `intergrax/rag/embedding/runtime/tokenizer_codec.py`
- `platform_proofs/scenarios/verified_product_identification/**` (multiple)
- `tests/unit/platform_proofs/scenarios/verified_product_identification/**` (multiple)

Mandatory certification gates were rerun against current HEAD.

## Frozen architecture model (static proof)

| Contract | Status |
|---|---|
| ExecutionRuntime sole lifecycle owner | YES (NPSC-3C/4/4.1/UE-10R gates green) |
| identity_authority sole mint owner | YES (UE-10R3 gate green) |
| ExecutionBoundary bind/reset authority | YES (NPSC-4.1 gate green) |
| StrategyExecutionRouter sole strategy owner | YES |
| HostTaskExecutionPort canonical host boundary | YES (NPSC-1 / UE-11GP green) |
| NexusLoop private orchestration backend only | YES (no Tier-3 `_orchestration_backend` access) |

## Legacy compatibility surface

| Symbol | Production status |
|---|---|
| `HarnessHostLegacyComposition` | ABSENT (`harness_host_runtime_compat.py` deleted) |
| `resolve_harness_host_nexus_loop_legacy` | ABSENT |
| `_legacy_composition` | ABSENT |

Gate: `test_npsc42_retired_legacy_compat_tokens_absent_from_production` — PASS

## Raw Nexus hermeticity

| Surface | Count |
|---|---|
| Tier-3 `host/factory.py` raw `NexusLoop` import | 0 |
| Tier-3 `_internal_composition` direct access | 0 |
| Tier-3 `_orchestration_backend` direct access | 0 |
| Auxiliary raw execution wiring | 0 (uses `resolve_harness_host_execution_terminal`, `HostTaskExecutionPort`) |

Production `_orchestration_backend` consumers (allowlisted):

1. `intergrax/applications/_shared/harness_host_composition.py` — composition owner / platform plugin bootstrap
2. `applications/local_workspace_application/model_runtime_proof/runtime.py` — internal proof root
3. `scripts/maintenance/check_harness_security_wiring.py` — maintenance verification
4. `scripts/maintenance/check_harness_reliability_wiring.py` — maintenance verification

Gate: `test_npsc42_orchestration_backend_access_confined_to_allowlist` — PASS

## Allowlist review

All four allowlisted paths exist; reasons remain valid; access is composition/internal only; no lifecycle or identity ownership.

## Tier-3 / plugin bootstrap — **FAIL**

| Factory | Platform bootstrap call | Status |
|---|---|---|
| `local_workspace_application` | `bootstrap_harness_host_platform(runtime)` | CORRECT |
| `research_application` | `bootstrap_harness_host_platform(runtime)` | CORRECT |
| `legal_application` | `bootstrap_nexus_platform(nexus_loop, …)` | **BROKEN** — `nexus_loop` undefined; `bootstrap_nexus_platform` not imported |
| `governed_contractor_application` | same | **BROKEN** |
| `dispute_sim_application` | same | **BROKEN** |
| `intergrax_assistant_application` | same | **BROKEN** |
| `attestation_demo` | same | **BROKEN** |
| `poc_template_application` | same | **BROKEN** |
| `lab_application` | same | **BROKEN** |

**Tier-3 direct `bootstrap_nexus_platform`:** 7 (required 0)

These factories import `bootstrap_harness_host_platform` but still invoke the retired direct Nexus bootstrap path with an undefined `nexus_loop` symbol — incomplete NPSC-4.2 migration; runtime `NameError` on factory entry.

## Scheduler / debug / trace (static + focused tests)

| Area | Result |
|---|---|
| Scheduler → `resolve_harness_host_execution_terminal` | PASS (static) |
| Debug execution → `HostTaskExecutionPort` | PASS (static) |
| `RunTraceStore` contract | PASS |
| `NexusObservabilityStores.trace_store: RunTraceStore` | PASS |
| `bootstrap_harness_host_platform` trace suppression | ABSENT (PASS) |

## Regression matrix

| Suite | Result | Notes |
|---|---|---|
| NPSC-3C gate | PASS | |
| NPSC-3E gate | PASS | |
| NPSC-3F gate | PASS | |
| NPSC-3G gate | PASS | |
| NPSC-4 gate | PASS | |
| NPSC-4.1 gate | PASS | |
| NPSC-4.2 gate | PASS | |
| NPSC-4.2-R1 trace contract | PASS | |
| UE-10R1–R4 gates | PASS | |
| UE-10R41 | KNOWN_PREEXISTING_FAIL | `decision_finalization_conformance.py` local `ThreadPoolExecutor` imports — unchanged |
| UE-11GP gate | PASS | |
| Execution + interaction (`tests/unit/runtime/execution/**`, `interactions/**`) | PASS | 671 passed |
| Core harness/composition (NPSC-1, NPSC-4.2, trace, scaffold, observability) | PASS | 70 passed |
| Expanded harness/composition (task_control, debug integration, startup guards) | **FAIL** | 15 failed / 105 run |

### Expanded harness failure causes (non-exhaustive)

- `test_task_control_host_composition.py` — `EffectiveProfileRevisionError` (test helpers omit durable profile persistence kwargs present in NPSC-1 gate fixtures)
- `test_application_startup_guards.py` — LKW registry projection `ctx.environment` None
- `test_debug_api_injected_trace_store_b09.py` — `NexusLoop.handle_task()` missing required `run_id`
- `test_debug_api.py` — `runtime_events` None

## Quality scan (NPSC-4.2 surface)

| Check | Result |
|---|---|
| NPSC-4.2 task-introduced prohibited patterns | NONE in `harness_host_composition.py` bootstrap |
| `bootstrap_harness_host_platform` `# type: ignore[arg-type]` | ABSENT |
| `bootstrap_harness_host_platform` `cast(` | ABSENT |
| Tier-3 factory `# type: ignore[arg-type]` on trace_store | PRESENT (9 factories) — **residual debt** |

## Residual debt classification

### NON-BLOCKING (if other gates pass)

- UE-10R41 local-import hygiene (pre-existing, unchanged)
- Tier-3 factory `# type: ignore[arg-type]` on trace_store passes (R1 corrected composition root only)
- Docker `runtime-context` vendored copies (out of production scope)

### BLOCKING (this certification)

- Seven Tier-3 factories still call `bootstrap_nexus_platform(nexus_loop)` — incomplete plugin bootstrap migration
- Expanded harness/composition regression not ALL PASS

### OUT-OF-SCOPE repository debt

- VPI / embedding intervening commits on HEAD
- Optional-dep collection errors in full `gate and not no_ci` local run (celery, langchain, etc.)

## Remote stability

Remote did not advance during certification.

## Final verdict

**NPSC-4.2-F: FAIL** — architecture gates and core NPSC-4.2 contracts hold, but Tier-3 plugin bootstrap migration is incomplete (7 broken factories) and expanded harness/composition regression is red. Freeze not granted.

## Next correction task

**NPSC-4.2-F1 — Tier-3 factory plugin bootstrap completion**

1. Replace `bootstrap_nexus_platform(nexus_loop, …)` with `bootstrap_harness_host_platform(runtime, trace_store=…)` in all seven remaining factories.
2. Remove obsolete `nexus_loop` references and unused imports.
3. Rerun Tier-3 factory integration tests and expanded harness/composition suite.
4. Re-execute NPSC-4.2-F certification on clean HEAD.
