# ADR-APP-002: EnvironmentSnapshot on STRICT task intake

| Field | Value |
|-------|-------|
| **Status** | Accepted · **implemented** (`APP-EVOL-1`) |
| **Date** | 2026-06-11 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §49.1.2 · plan `APP-EVOL-1` |

## Context

Tier-3 architecture §49 requires immutable **EnvironmentSnapshot** materialization at deploy or task intake so STRICT production tasks execute against a resolved profile fingerprint, not mutable on-disk YAML.

`ApplicationEnvironmentState.profile_snapshot_id` existed as a field but was seeded with `profile_id` until APP-EVOL-1.

## Decision

1. Introduce typed **`EnvironmentSnapshot`** (`environment_snapshot.v1`) with stable digests for manifest, roster, profile, graph spec, and org envelope.
2. Attach **`EnvironmentSnapshotMiddleware`** (priority 35) on `build_harness_host_runtime` to capture on `BEFORE_TASK_INTAKE`.
3. Persist snapshot on `Task.metadata` via Nexus lifecycle merge/persist (wire key in `TaskMetadataKey.ENVIRONMENT_SNAPSHOT`).
4. Seed **`ApplicationEnvironmentState.profile_snapshot_id`** from captured snapshot before env-state middleware (priority 40).
5. **STRICT** tasks MUST receive a digest-based `profile_snapshot_id`; non-STRICT hosts MAY fall back to `profile_id`.

## Consequences

### Positive

- Auditable intake fingerprint for replay, diff, and incident comparison.
- Single harness path — all reference hosts via `build_harness_host_runtime`.

### Negative

- Small Tier-1 lifecycle change to persist snapshot metadata key.
- Profile digest recomputed per intake (acceptable until deploy-time snapshot cache in APP-EVOL-6).

## Compliance

- Tier boundaries: wire keys in `intergrax/runtime/task/task_metadata_keys.py`; contracts in `intergrax/applications/contracts/`.
- No Nexus fork; snapshot is metadata only.

## Implementation notes

- `intergrax/applications/contracts/environment_snapshot.py`
- `intergrax/applications/_shared/environment_snapshot_wiring.py`
- `uv run pytest tests/unit/applications/test_environment_snapshot_wiring.py -q`
