---
id: IJ-2026-06-11-035
date: 2026-06-11
tiers:
  - tier-1
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-8
  - APP-PROD-8
status: completed
commit: pending
adr: none — extends existing §20–§21 lifecycle wiring; no new primitives
---

# Sprint 10 — shadow/sandbox env-state refs and factory lifespan cleanup

## Operator request

Continue Tier-3 ACP-FINISH sprint queue: implement APP-CON-8 (isolation refs in `ApplicationEnvironmentState` + lifespan cleanup) and APP-PROD-8 (`check_workspace_cleanup` CI gate + factory teardown tests).

## Summary

- `workspace_cleanup_wiring.py` — `sync_isolation_refs_for_hook`, `make_workspace_cleanup_lifespan`, `build_factory_lifespans`, `apply_factory_lifespans`.
- `application_environment_state_middleware.py` mirrors shadow/sandbox handles on lifecycle hooks.
- `workspace_cleanup.py` — `clear_isolation_refs_in_task_env_state`; `task_finisher.py` clears refs after per-task cleanup.
- `ShadowWorkspaceManager` / `SandboxSessionManager` — `dispose_all_active()` for host shutdown.
- All seven reference host factories use `build_factory_lifespans`.
- `scripts/check_workspace_cleanup.py` wired into `check_application_production_gates.py` (APP-PROD-8).

## Project impact

Product hosts now purge lingering shadow/sandbox sessions on shutdown and expose active isolation handles in host-visible env state. Closes §20–§21 lifecycle cleanup gap; unblocks APP-EVOL-1 (EnvironmentSnapshot on intake).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §20–§21 · APP-PROD-8 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-CON-8 · APP-PROD-8 · §6.2y step 7 |
| ADR | none — reuses existing isolation managers |

## Changed artifacts

- `intergrax/applications/_shared/workspace_cleanup_wiring.py` — lifespan + env-state sync helpers
- `intergrax/applications/_shared/application_environment_state_middleware.py` — isolation ref sync
- `intergrax/runtime/nexus/orchestration/workspace_cleanup.py` — clear env refs post-task
- `intergrax/runtime/nexus/orchestration/task_finisher.py` — invoke clear after cleanup
- `intergrax/runtime/workspace/manager.py` · `intergrax/runtime/sandbox/manager.py` — `dispose_all_active`
- `applications/*/host/factory.py` (×7) — `build_factory_lifespans`
- `scripts/check_workspace_cleanup.py` · `scripts/check_application_production_gates.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_workspace_cleanup_wiring.py \
  tests/unit/runtime/nexus/orchestration/test_workspace_cleanup_env_state.py \
  tests/unit/scripts/test_check_workspace_cleanup.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
python scripts/check_implementation_journal.py
```

Result: pass (9 tests).

## Risks and follow-ups

- Isolation IDs propagate via hook `extra` / execution structured_data — step-kernel auto-merge into `runtime_state` deferred.
- APP-EVOL-1 next in §6.2y queue.
