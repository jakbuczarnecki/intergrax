# ADR-AGENT-008 — Durable registry projection rehydration authority

**Status:** Accepted  
**Date:** 2026-09-06  
**Domain:** Agent Distribution / AP-10 / EA-03

## Context

Enterprise acceptance requires cold process restart to restore traffic-serving execution without any runtime object from the previous process. Durable SQLite stores already preserved installations, bindings, roster snapshots, runtime revisions, materializations, locks, and the serving pointer — but `MaterializedRegistryProjection` lived only in process-local `RuntimeRegistryProjectionStore`.

The prior production host rule stated startup-time reprojection is forbidden. That correctly blocked rebuilding from mutable desired state, but it left no canonical path to reconstruct the traffic-serving projection after restart.

## Decision

Adopt explicit separation:

```text
DURABLE RUNTIME AUTHORITY ≠ PROCESS-LOCAL RUNTIME OBJECT
```

1. **Persist** an immutable `RuntimeRegistryProjectionDescriptor` at activation time (before serving commit), keyed by `runtime_revision_id`.
2. **Pin** revision-bound manifest/build-context identity in the descriptor (Option B) because release-bound manifest is not yet a standalone durable artifact.
3. **Rehydrate** on process composition startup via `RuntimeRegistryProjectionRehydrator` using the same canonical projection builder path as activation.
4. **Fail closed** when serving pointer exists but descriptor is missing, corrupt, or mismatched with canonical revision authority.

Lifecycle ownership remains unchanged: rehydration is not activation, not install/bind, and not desired-state recomputation.

## Rejected alternatives

| Alternative | Why rejected |
|-------------|--------------|
| Serialize live `MaterializedRegistryProjection` / agent objects | Violates tier boundaries; not revision-safe; couples execution state to durability |
| Startup reprojection from current manifest/roster | Rebuilds from mutable desired state; breaks historical serving authority |
| Implicit lazy rebuild on first request | Hides failure until traffic; weak operator readiness signal |

## Consequences

- `SERVING(N) ⇒ descriptor(N)` is enforced before activation commit.
- SQLite adapter adds `projection_descriptors` table behind `RuntimeRegistryProjectionDescriptorStore`.
- `directory_content_digest` skips non-authoritative cache dirs (e.g. `__pycache__`) so execution side effects do not invalidate immutable artifact identity.
- Composition roots call rehydration explicitly (fail-early startup readiness).

## References

- [`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) — Activation-time projection authority vs restart-time rehydration
- `intergrax/applications/_shared/registry_projection_descriptor.py`
- `intergrax/applications/_shared/registry_projection_rehydrator.py`
