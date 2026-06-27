---
id: IJ-2026-06-11-036
date: 2026-06-11
tiers:
  - tier-1
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-1
status: completed
commit: pending
adr: docs/adr/entries/2026-06-12/ADR-APP-002.md
---

# Sprint 11 — EnvironmentSnapshot on STRICT task intake

## Operator request

Execute the next Tier-3 application architecture sprint: APP-EVOL-1 — materialize `EnvironmentSnapshot` on intake and record `profile_snapshot_id` on STRICT tasks.

## Summary

- `EnvironmentSnapshot` contract (`environment_snapshot.v1`) with stable digests.
- `EnvironmentSnapshotMiddleware` (priority 35) captures snapshot on `BEFORE_TASK_INTAKE`.
- Nexus lifecycle persists snapshot on `Task.metadata` via `TaskMetadataKey.ENVIRONMENT_SNAPSHOT`.
- `ApplicationEnvironmentState` seeds `profile_snapshot_id` from captured digest (STRICT requires fingerprint; non-STRICT may fall back to `profile_id`).
- ADR-APP-002 documents mandatory STRICT intake fingerprint.

## Project impact

STRICT production tasks now carry an auditable environment fingerprint at intake — foundation for replay, diff (`APP-EVOL-6`), and incident comparison per architecture §49.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.1.2 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-1 · §6.2y step 8 |
| ADR | `docs/adr/entries/2026-06-12/ADR-APP-002.md` |

## Changed artifacts

- `intergrax/applications/contracts/environment_snapshot.py` — snapshot contract
- `intergrax/applications/_shared/environment_snapshot_wiring.py` — digest + capture
- `intergrax/applications/_shared/environment_snapshot_middleware.py` — intake middleware
- `intergrax/runtime/hooks/nexus_lifecycle_hooks.py` — snapshot persist
- `intergrax/runtime/task/task_metadata_keys.py` — wire keys
- `intergrax/applications/_shared/harness_host_runtime.py` — wiring

## Verification

```bash
uv run pytest tests/unit/applications/test_environment_snapshot_wiring.py \
  tests/unit/applications/test_application_environment_state_lifecycle.py -q
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Per-intake profile digest recompute — deploy-time snapshot cache deferred to APP-EVOL-6.
- APP-OPS-1 next in §6.2y queue.
