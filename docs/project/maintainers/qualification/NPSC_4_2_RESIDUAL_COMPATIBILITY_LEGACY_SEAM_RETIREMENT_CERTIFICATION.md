# NPSC-4.2 — Residual Compatibility & Legacy Seam Retirement Certification

**Verdict:** PASS  
**Date:** 2026-09-08  
**Branch:** `development`

## 1. Starting residual seams

| Seam | Location | Purpose |
|---|---|---|
| `HarnessHostLegacyComposition` | `harness_host_runtime_compat.py` | Stored raw `NexusLoop` on `HarnessHostRuntime` |
| `resolve_harness_host_nexus_loop_legacy` | `harness_host_runtime_compat.py` | Cross-component private access to `_legacy_composition.nexus_loop` |
| Auxiliary wiring via raw Nexus | `harness_host_auxiliary_wiring.py` | Plugin bootstrap, scheduler terminal, debug composition |
| Tier-3 factory bootstrap | `applications/*/host/factory.py` | `bootstrap_nexus_platform(nexus_loop)` after legacy resolve |

## 2. Consumer inventory (before)

| File | Symbol | Why raw Nexus | Members used | Lifecycle/identity | Canonical port possible | Narrow capability | Classification |
|---|---|---|---|---|---|---|---|
| `harness_host_auxiliary_wiring.py` | legacy resolver | plugin/scheduler/debug | `execution_terminal`, full loop | NO | partial | YES | PLUGIN_BOOTSTRAP / TERMINAL_SIGNAL / DEBUG_INTERNAL |
| `task_control_wiring.py` | legacy resolver | task cancel terminal | `execution_terminal` | NO | YES | YES | TERMINAL_SIGNAL |
| `diagnostic_read_wiring.py` | legacy resolver | runtime events fallback | `runtime_event_store` | NO | YES | YES | OBSERVABILITY |
| `auditability_health_wiring.py` | legacy resolver | runtime events fallback | `runtime_event_store` | NO | YES | YES | OBSERVABILITY |
| `acp_session_host_wiring.py` | legacy resolver | ACP session gate | `peek_decision_flow_gate()` | NO | NO | YES | ORCHESTRATION_INTERNAL |
| `applications/*/host/factory.py` (9) | legacy resolver | platform/plugins bootstrap | `nexus_loop`, `event_bus` | NO | partial | YES | PLUGIN_BOOTSTRAP |
| `lab_fastapi.py` | legacy resolver | lab routes + plugins | `registry`, full loop | NO | YES | YES | PLUGIN_BOOTSTRAP / OBSERVABILITY |
| `scripts/maintenance/check_harness_*.py` | legacy resolver | assembly verification | middleware, checkpoint | NO | partial | YES | OBSERVABILITY |
| tests (multiple) | legacy resolver | behavioral assertions | assorted internals | NO | N/A | N/A | TEST_ONLY |

**Unexpected Tier-3 execution consumers:** NONE

## 3. Before / after architecture

**Before**

```text
consumer → resolve_harness_host_nexus_loop_legacy(runtime) → NexusLoop → arbitrary internals
```

**After**

```text
consumer → narrow typed capability (harness_host_composition) → internal composition owner → Nexus (private)
```

`HarnessHostRuntime` now owns `_internal_composition: HarnessHostInternalComposition` with explicit fields:

- `execution_terminal`
- `event_bus`
- `decision_flow_gate`
- `middleware_pipeline`
- `lifecycle_hook_coordinator`
- `plugin_surface`
- `runtime_event_persistence`
- `_orchestration_backend` (composition-root only; not exposed via legacy resolver)

## 4. Migration matrix

| File | Old dependency | New dependency | Classification | Parity |
|---|---|---|---|---|
| `harness_host_auxiliary_wiring.py` | raw Nexus resolver | `resolve_harness_host_execution_terminal`, `bootstrap_harness_host_platform`, `host_execution` for debug | TERMINAL_SIGNAL / DEBUG_INTERNAL | PASS |
| `task_control_wiring.py` | raw Nexus resolver | `resolve_harness_host_execution_terminal` | TERMINAL_SIGNAL | PASS |
| `diagnostic_read_wiring.py` | raw Nexus fallback | `resolve_harness_host_runtime_event_persistence` | OBSERVABILITY | PASS |
| `auditability_health_wiring.py` | raw Nexus fallback | `resolve_harness_host_runtime_event_persistence` | OBSERVABILITY | PASS |
| `acp_session_host_wiring.py` | raw Nexus gate | `resolve_harness_host_decision_flow_gate` | ORCHESTRATION_INTERNAL | PASS |
| Tier-3 factories (9) | raw Nexus bootstrap | `bootstrap_harness_host_platform` / `bootstrap_harness_host_application_plugins` | PLUGIN_BOOTSTRAP | PASS |
| `lab_fastapi.py` | raw Nexus routes/plugins | `runtime.registry`, `bootstrap_harness_host_platform` | PLUGIN_BOOTSTRAP | PASS |
| `debug/app.py` | Nexus-only HITL/intake fallback | `host_execution` canonical path | DEBUG_INTERNAL | PASS |

## 5. Retained exceptions

| Item | Status |
|---|---|
| `resolve_harness_host_nexus_loop_legacy` | **REMOVED** |
| `HarnessHostLegacyComposition` | **REMOVED** |
| `_legacy_composition` | **REMOVED** |
| `_orchestration_backend` inside `HarnessHostInternalComposition` | **RETAINED** — composition-root private backend for plugin platform bootstrap only; not reachable via retired public resolver |

## 6. Lifecycle ownership proof

- `ExecutionRuntime` remains sole lifecycle owner (NPSC-3C/4/4.1 gates green)
- No new lifecycle abstraction introduced
- Scheduler receives `execution_terminal` only; lifecycle ownership unchanged

## 7. Identity ownership proof

- `identity_authority.py` remains sole mint authority (NPSC-4 gate green)
- No new mint/bind outside `ExecutionBoundary`

## 8. Raw-Nexus surface proof

- Retired resolver removed from production
- Tier-3 `host/` factories: **0** legacy resolver imports
- Raw `NexusLoop` imports in shared wiring limited to composition-owner allowlist (gate `test_npsc42_raw_nexus_imports_remain_composition_owner_allowlist`)

## 9. Tier-3 surface proof

Gate: `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py`

- `test_npsc42_tier3_host_factories_have_no_legacy_compat_imports` — PASS

## 10. Regression evidence

| Suite | Result | Notes |
|---|---|---|
| NPSC-4.2 gate | PASS (5) | |
| NPSC-4 / 4.1 gates | PASS | |
| NPSC-3C / 3E / 3F / 3G gates | PASS | |
| NPSC-1 / NPSC-2 / NPSC-3B gates | PASS | |
| UE-10R1–R4 gates | PASS | |
| UE-10R41 | FAIL (pre-existing) | `decision_finalization_conformance.py` local import — unrelated to NPSC-4.2 |
| UE-11GP gate | PASS (5) | |
| Focused compatibility (gate-marked, not no_ci) | PASS (76) | |

## 11. Remaining debt

- `test_ue_10r41_execution_import_hygiene_gate` pre-existing failure (not introduced by NPSC-4.2)
- Docker `runtime-context` vendored copies not updated (out of NPSC-4.2 production scope)

## 12. Final verdict

**NPSC-4.2: PASS** — legacy compatibility resolver and composition seam retired; consumers migrated to narrow typed harness host capabilities; architecture gates enforce Tier-3 and production surface closure.
