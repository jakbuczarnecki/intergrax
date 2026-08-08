---
id: IJ-2026-06-11-027
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-3
status: completed
commit: pending
adr: none — middleware composition on existing HookContext contract; no new platform semantics
---

# Sprint 2 APP-CON-3 — environment state lifecycle sync on Nexus hooks

## Operator request

Continue the Tier-3 sprint backlog after APP-PROD-9: implement APP-CON-3 so Nexus lifecycle hooks automatically maintain `app_env_state.v1` (phase, budget limits, HITL posture) for ApplicationHost consumers.

## Summary

Added `ApplicationEnvironmentStateMiddleware` (priority 40, before `ApplicationHost` at 50) wired from `build_harness_host_runtime`. The middleware seeds and updates `ApplicationEnvironmentState` on every lifecycle hook. `NexusLifecycleHookCoordinator` now merges and persists `app_env_state.v1` through `task.metadata` across hook invocations. Integration tests cover phase progression and HITL health transitions.

## Project impact

Tier-3 hosts receive a consistent, task-scoped environment state on every Nexus lifecycle hook without custom host code. Unblocks ACP-TOK-3 host reactions and partial §43 budget visibility (limits seeded from `RunBudget`; live token rollups remain ACP-TOK-1/2).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §42 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` APP-CON-3 |
| ADR | none — extends existing hook middleware pattern |

## Changed artifacts

- `intergrax/applications/_shared/application_environment_state_middleware.py` — lifecycle sync middleware
- `intergrax/applications/_shared/application_host_wiring.py` — `apply_application_environment_state_wiring`
- `intergrax/applications/_shared/harness_host_runtime.py` — mount env-state middleware on all hosts
- `intergrax/applications/contracts/environment_state.py` — `APP_ENV_STATE_RUNTIME_KEY` constant
- `intergrax/runtime/hooks/nexus_lifecycle_hooks.py` — task.metadata merge/persist for env state
- `tests/unit/applications/test_application_environment_state_lifecycle.py` — integration tests
- `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — APP-CON-3 **Done**

## Verification

```bash
uv run pytest tests/unit/applications/test_application_environment_state_lifecycle.py tests/unit/applications/test_application_host_wiring.py tests/unit/applications/test_environment_state_and_artifacts.py -q
```

Result: pass (8 tests).

## Risks and follow-ups

- Live token metering (`agent_tokens_total`, `environment_tokens_total`) awaits ACP-TOK-1/2; middleware only seeds `environment_tokens_limit` from `RunBudget`.
- APP-CON-5 (hook timeout / error→BLOCK) and APP-CON-6 (artifact bundle) remain open in sprint backlog.
