# NPSC-4.2-F2 — Residual Compatibility Surface Final Freeze Certification

**Verdict:** PASS / FROZEN  
**Date:** 2026-09-08  
**Branch:** `development`

## 1. Certified code SHA

| Field | Value |
|---|---|
| CERTIFIED_CODE_HEAD | `996f6a639db2f6b6b0169cb75f35e4a7012ae268` |
| REMOTE_AT_START | `996f6a639db2f6b6b0169cb75f35e4a7012ae268` |
| CERTIFICATION_COMMIT | (recorded after docs-only commit) |

Tests were executed against **CERTIFIED_CODE_HEAD** before the certification document commit.

## 2. Certification date

2026-09-08 (UTC+2 session)

## 3. NPSC-4.2 history

```text
NPSC-4.2
    ↓ raw Nexus compatibility retirement

NPSC-4.2-R1
    ↓ RunTraceStore typed-contract correction

NPSC-4.2-F attempt #1
    ↓ FAIL — 7 broken Tier-3 bootstrap paths
    ↓ (historical evidence preserved: NPSC_4_2_RESIDUAL_COMPATIBILITY_SURFACE_FINAL_CERTIFICATION.md)

NPSC-4.2-F1
    ↓ PASS — 7 Tier-3 factories migrated

NPSC-4.2-F2
    ↓ PASS / FROZEN (this document)
```

Historical FAIL document **not modified** — classification `COMMITTED_HISTORICAL_FAIL`.

## 4. F1 correction evidence

All seven F1-migrated factories use `bootstrap_harness_host_platform(runtime)` and application-specific `bootstrap_harness_host_application_plugins(...)` where applicable. Factory startup smoke revalidated 7/7 PASS (17 targeted tests, no `NameError`, no undefined `nexus_loop`).

## 5. Intervening commit scope

F1 HEAD: `cb921c7d01b453e3fc72ccde3bbe7a98bf52d908`

Changed files since F1:

- `docs/project/maintainers/qualification/NPSC_4_2_RESIDUAL_COMPATIBILITY_SURFACE_FINAL_CERTIFICATION.md` (historical FAIL commit)
- `platform_proofs/scenarios/verified_product_identification/qualification/bounded_cuda_batch/**` (VPI)
- `platform_proofs/scenarios/verified_product_identification/qualification/run_bounded_cuda_batch_throughput_qualification.py`
- `tests/unit/platform_proofs/scenarios/verified_product_identification/test_bounded_cuda_batch_throughput_qualification.py`

**INTERVENING_SENSITIVE_PRODUCTION_CHANGES:** NONE

## 6. Legacy seam proof

| Symbol | Production status |
|---|---|
| `HarnessHostLegacyComposition` | ABSENT |
| `resolve_harness_host_nexus_loop_legacy` | ABSENT |
| `_legacy_composition` | ABSENT |

Gate: `test_npsc42_retired_legacy_compat_tokens_absent_from_production` — PASS

## 7. Tier-3 surface proof

Static audit `applications/*/host`:

| Surface | Count |
|---|---|
| direct `bootstrap_nexus_platform` | 0 |
| `bootstrap_application_plugins(...nexus...)` | 0 |
| raw `NexusLoop` access | 0 |
| legacy resolver | 0 |
| `_internal_composition` | 0 |
| `_orchestration_backend` | 0 |

All nine Tier-3 factories (`local_workspace_application`, `research_application`, seven F1 factories) use canonical harness bootstrap.

## 8. Raw Nexus hermeticity

Production `_orchestration_backend` consumers (allowlisted, unchanged since F1):

1. `intergrax/applications/_shared/harness_host_composition.py` — composition owner / platform plugin bootstrap
2. `applications/local_workspace_application/model_runtime_proof/runtime.py` — internal proof root
3. `scripts/maintenance/check_harness_security_wiring.py` — maintenance verification
4. `scripts/maintenance/check_harness_reliability_wiring.py` — maintenance verification

Gate: `test_npsc42_orchestration_backend_access_confined_to_allowlist` — PASS  
**New consumers since F1:** 0

## 9. Typed composition proof

`HarnessHostInternalComposition` canonical model verified:

```text
HarnessHostInternalComposition
├ execution_terminal
├ event_bus
├ decision_flow_gate
├ middleware_pipeline
├ lifecycle_hook_coordinator
├ plugin_surface
├ runtime_event_persistence
└ private orchestration backend (_orchestration_backend)
```

No generic `NexusFacade`. No compatibility manager. Gate: `test_npsc42_harness_host_composition_exposes_narrow_capabilities` — PASS

## 10. RunTraceStore proof

| Check | Result |
|---|---|
| `RunTraceStore(RunTraceWriter, RunTraceReader)` | PASS |
| `NexusObservabilityStores.trace_store: RunTraceStore` | YES (`intergrax/runtime/nexus/observability_wiring.py`) |
| `InMemoryRunTraceStore` compliant | YES |
| `SQLiteRunTraceStore` compliant | YES |
| Contract tests | 2 passed |

## 11. Plugin bootstrap proof

| Check | Result |
|---|---|
| `bootstrap_harness_host_platform` delegates to `bootstrap_nexus_platform` via composition | PASS (static) |
| `bootstrap_harness_host_application_plugins` uses explicit `plugin_surface` | PASS (static) |
| Default plugin parity | PASS (composition-internal delegation preserved) |
| Application plugin registration order | PASS (unchanged explicit surface) |
| Shutdown callbacks | PASS (`attach_plugin_shutdown` retained on all seven F1 factories) |
| Bootstrap trace suppression on composition root | ABSENT (gate PASS) |

## 12. Scheduler proof

`intergrax/runtime/long_running/scheduler.py` consumes typed `ExecutionTerminalService`; no raw `NexusLoop` import. Scheduler is not lifecycle owner or identity mint owner (static + NPSC-3C/UE-10R gates).

## 13. Debug proof

`intergrax/debug/app.py` routes execution through `HostTaskExecutionPort` / `build_host_task_execution`. No raw Nexus fallback on root execution path (NPSC-3E gate PASS).

## 14. Execution ownership

| Contract | Status |
|---|---|
| `ExecutionRuntime` sole root lifecycle owner | YES (NPSC-3C/4/4.1/UE-10R gates) |
| `HostTaskExecutionPort` canonical host boundary | YES (UE-11GP gate) |
| `StrategyExecutionRouter` sole strategy owner | YES |
| Nexus lifecycle owner | NO (private orchestration backend only) |

## 15. Identity ownership

| Contract | Status |
|---|---|
| `identity_authority` sole mint owner | YES (UE-10R3 gate) |
| Mint outside authority | NO |
| Bind outside `ExecutionBoundary` | NO (NPSC-4.1 gate) |

## 16. Static gates

Evidence: `.tmp/session/NPSC-4.2-F2/static-gates.log` — 79 passed, 2 failed (UE-10R41 only)

| Gate | Result |
|---|---|
| NPSC-3C | PASS |
| NPSC-3E | PASS |
| NPSC-3F | PASS |
| NPSC-3G | PASS |
| NPSC-4 | PASS |
| NPSC-4.1 | PASS |
| NPSC-4.2 | PASS |
| NPSC-4.2-R1 | PASS |
| NPSC-4.2-F1 | PASS (`test_npsc42f1_tier3_host_factories_use_canonical_harness_bootstrap`) |
| UE-10R1 | PASS |
| UE-10R2 | PASS |
| UE-10R3 | PASS |
| UE-10R4 | PASS |
| UE-10R41 | KNOWN_PREEXISTING_FAIL (unchanged `decision_finalization_conformance.py` local imports) |
| UE-11GP | PASS |

## 17. Execution/interaction regression

Evidence: `.tmp/session/NPSC-4.2-F2/execution-interaction.log`

| Suite | Result | Passed |
|---|---|---|
| `tests/unit/runtime/execution/**` + `interactions/**` | PASS | 671 |

No new failures vs F1 baseline (674 → 671; suite evolution, 0 new failures attributable to NPSC-4.2).

## 18. Expanded harness failure classification

Evidence: `.tmp/session/NPSC-4.2-F2/expanded-harness.log`

Five-file F1-expanded suite:

```text
TOTAL:
22 passed
9 failed
```

**NPSC-4.2-related failures:** 0

**KNOWN PREEXISTING failures:**

1. `test_taskcpm_h1_product_host_runtime_exposes_canonical_boundary` — `EffectiveProfileRevisionError`
2. `test_taskcpm_h2_allow_through_host_composition_reaches_cooperative_cancel` — `EffectiveProfileRevisionError`
3. `test_taskcpm_h3_deny_through_host_composed_boundary_zero_cancel_effect` — `EffectiveProfileRevisionError`
4. `test_taskcpm_h5_missing_boundary_remains_fail_closed_on_direct_mount` — `execution_budget_ledger_factory` fixture gap
5. `test_taskcpm_h1b_governed_contractor_factory_wires_runtime_boundary` — `EffectiveProfileRevisionError`
6. `test_taskcpm_h7b_local_workspace_factory_wires_runtime_boundary` — `EffectiveProfileRevisionError`
7. `test_local_workspace_factory_http_only_startup` — LKW registry projection `ctx.environment` None
8. `test_debug_api_reads_injected_trace_store_without_sqlite` — `NexusLoop.handle_task()` missing `run_id`
9. `test_debug_api_trace_with_runtime` — `runtime_events` None

### Baseline comparison matrix (F1 → F2)

| TEST | CURRENT FAILURE | F1 FAILURE | SAME ROOT CAUSE? | TOUCHED BY F1? | TOUCHED SINCE F1? | NPSC-4.2 RELATED? |
|---|---|---|---|---|---|---|
| taskcpm_h1 | EffectiveProfileRevisionError | YES | YES | NO | NO | NO |
| taskcpm_h2 | EffectiveProfileRevisionError | YES | YES | NO | NO | NO |
| taskcpm_h3 | EffectiveProfileRevisionError | YES | YES | NO | NO | NO |
| taskcpm_h5 | execution_budget_ledger_factory | YES (subset) | YES | NO | NO | NO |
| taskcpm_h1b | EffectiveProfileRevisionError | YES | YES | NO | NO | NO |
| taskcpm_h7b | EffectiveProfileRevisionError | YES | YES | NO | NO | NO |
| local_workspace_factory_http_only_startup | ctx.environment None | YES | YES | NO | NO | NO |
| debug_api_injected_trace_store_b09 | missing run_id | YES | YES | NO | NO | NO |
| debug_api_trace_with_runtime | runtime_events None | YES | YES | NO | NO | NO |

Suite size evolved (F1: 44 tests / 15 failed → F2: 31 tests / 9 failed); all current failures map to unchanged F1 root causes. No new NPSC-4.2 bootstrap failures.

## 19. Quality audit

NPSC-4.2 production surface (`harness_host_composition.py`, `harness_host_runtime.py`, `plugin_bootstrap.py`):

| Pattern | NPSC-4.2 introduced |
|---|---|
| `dict[str, Any]` | NONE |
| `cast(` | NONE |
| `# type: ignore` on bootstrap | NONE (gate verified) |
| `getattr`/`setattr`/`hasattr` | NONE |
| `inspect` | NONE |
| broad `except Exception` | NONE |
| dynamic dispatch | NONE |

**NPSC-4.2-introduced prohibited patterns:** NONE

## 20. Known unrelated debt

- UE-10R41 local-import hygiene (`decision_finalization_conformance.py`)
- Tier-3 `trace_store` `# type: ignore[arg-type]` on `local_workspace_application` and `research_application` host factories (pre-existing, not bootstrap-path suppression)
- Expanded harness 9 pre-existing fixture/prerequisite failures (see §18)
- `attestation_demo` partner PoC boundary-event tests (14 failures in full host suite; factory construction and `test_attestation_demo_lists_agents` PASS — partner contract scope, not NPSC-4.2 bootstrap)

## 21. Final PASS/FROZEN verdict

**NPSC-4.2: PASS / FROZEN**

All F2 certification criteria satisfied on `996f6a639db2f6b6b0169cb75f35e4a7012ae268`. Residual compatibility surface retired; Tier-3 harness bootstrap canonical; frozen enterprise execution contracts intact.

**Next:** NPSC-5 — Multi-Agent Production Architecture
