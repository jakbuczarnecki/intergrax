---
id: IJ-2026-06-11-033
date: 2026-06-11
tiers:
  - tier-1
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-5
status: completed
commit: pending
adr: none — enforcement in existing MiddlewarePipeline contract
---

# Sprint 8 APP-CON-5 — hook timeout, error→BLOCK, audit events

## Operator request

Continue Tier-3 application architecture sprint queue with APP-CON-5: enforce hook wall-time limits, map uncaught exceptions to BLOCK with `hook_error`, and emit audit runtime events on non-ALLOW hook results.

## Summary

Added `hook_runtime_guard.py` and extended `MiddlewarePipeline` with `configure_hook_runtime`. Each middleware and registry invocation runs under `asyncio.wait_for` when `ReliabilityProfile.middleware_hook_timeout_seconds` is set (0.25s on `product_defaults`). Violations emit `HOOK_TIMEOUT`, `HOOK_ERROR`, or `HOOK_BLOCKED` events. `apply_hook_runtime_guard_wiring` configures the pipeline from `build_harness_host_runtime`. Hook registry failures now use `hook_error:` reason prefix.

## Project impact

STRICT product hosts get deterministic fail-closed hook semantics — slow or failing `ApplicationHost` callbacks cannot stall Nexus intake. §32.6.5 timeout/error/audit rows are implementation-complete for Tier-3 wiring.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §32.6.5 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-CON-5 · §6.2y step 5 |

## Changed artifacts

- `intergrax/runtime/middleware/hook_runtime_guard.py` — guard + audit emit (new)
- `intergrax/runtime/middleware/pipeline.py` — guarded middleware/registry invocation
- `intergrax/runtime/events/runtime_event.py` — `HOOK_*` event types
- `intergrax/runtime/hooks/hook_registry.py` — `hook_error` reason prefix
- `intergrax/applications/contracts/environment_profile.py` — `middleware_hook_timeout_seconds`
- `intergrax/applications/_shared/application_host_wiring.py` — `apply_hook_runtime_guard_wiring`
- `intergrax/applications/_shared/harness_host_runtime.py` — wiring call
- `intergrax/harness/hooks.py` — preserve hook runtime on pipeline merge
- `tests/unit/runtime/middleware/test_hook_runtime_guard.py`
- `tests/unit/applications/test_application_hook_runtime.py`

## Verification

```bash
uv run pytest tests/unit/runtime/middleware/test_hook_runtime_guard.py tests/unit/applications/test_application_hook_runtime.py tests/unit/applications/test_application_environment_state_lifecycle.py -m gate -q
```

Result: pass (10 tests).

## Risks and follow-ups

- APP-CON-6: `RunArtifactBundle` on `ApplicationRunSummary.metadata`.
- Lab hosts with custom slow hooks may need higher `middleware_hook_timeout_seconds`.
