---
id: IJ-2026-06-11-034
date: 2026-06-11
tiers:
  - tier-1
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-6
status: completed
commit: pending
adr: none — rollup on existing RunArtifactBundle contract
---

# Sprint 9 APP-CON-6 — RunArtifactBundle on ApplicationRunSummary

## Operator request

Continue Tier-3 application architecture sprint queue with APP-CON-6: attach `RunArtifactBundle` rollup to Plane A `ApplicationRunSummary` metadata on task completion.

## Summary

Added `run_artifact_bundle_builder.py` to collect staged application artifacts, shadow workspace files, and sandbox outputs before isolation cleanup. `build_nexus_task_result` now materializes the bundle, nests it under `ApplicationRunSummary.metadata[run_artifact_bundle.v1]`, and mirrors the key on `TaskResult.metadata`. Introduced `stage_application_artifact()` helper and metadata constants `run_artifact_bundle.v1` / `application_artifacts.v1`.

## Project impact

Operators discover task artifacts via Plane A summary without scanning host filesystems. Closes §26/§48 rollup gap; enables APP-CON-8 shadow/sandbox cleanup integration to reference bundle refs.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §26 · §48 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-CON-6 · §6.2y step 6 |

## Changed artifacts

- `intergrax/applications/_shared/run_artifact_bundle_builder.py` — builder + staging helper (new)
- `intergrax/runtime/nexus/orchestration/task_finisher.py` — bundle before cleanup
- `intergrax/contracts/application_run_summary.py` — `metadata` field
- `intergrax/applications/contracts/application_artifacts.py` — metadata key constants
- `intergrax/runtime/task/task_metadata_keys.py` — `RUN_ARTIFACT_BUNDLE`
- `tests/unit/applications/test_run_artifact_bundle_builder.py`
- `tests/unit/runtime/nexus/orchestration/test_task_finisher_artifact_bundle.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_run_artifact_bundle_builder.py tests/unit/runtime/nexus/orchestration/test_task_finisher_artifact_bundle.py -q
```

Result: pass (3 tests).

## Risks and follow-ups

- APP-CON-8: shadow/sandbox lifespan cleanup should retain bundle URIs when `delete_on_task_complete=false`.
- Hosts should call `stage_application_artifact()` from `ApplicationHost` hooks for business exports.
