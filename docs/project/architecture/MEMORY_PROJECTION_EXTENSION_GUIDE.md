# Memory projection extension guide

**Canonical architecture:** [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md)

Projections synchronize **derived** state after canonical UserProfile mutations.

## Contract

Implement `UserProfileMemoryProjection` (`intergrax/memory/contracts/memory_lifecycle.py`):

- `upsert_memory_entry(context, entry)`
- `delete_memory_entries(context, entry_ids)`
- `reconcile(context)` → `MemoryProjectionReconciliationResult`

## Required rules

1. Accept **trusted typed context** — `UserProfileMemoryProjectionContext` / `UserProfileMemoryReconciliationContext`.
2. **Never construct `RequestIdentity`** from string ids inside the adapter.
3. **Canonical source remains UserProfile** — your projection is not authority.
4. **Failure must surface to lifecycle** — do not swallow errors; coordinator records projection evidence.
5. **Reconciliation must be supported** — repair from `authoritative_active_entry_ids` and profile snapshot.

Use factory helper `user_profile_memory_projection_context(identity, user_id, …)` from contracts — do not rebuild identity.

## Reference implementation

`EntityIndexerUserProfileMemoryProjection` (`applications/_shared/entity_user_profile_memory_projection.py`):

- Passes `context.identity` to `EntityMemoryIndexer`
- Reconcile compares `entity_capability.get_entity` revision vs canonical `entry.revision`

LTM vector: `UserProfileLtmVectorProjection` (`user_profile_ltm_vector_projection.py`).

## Wiring

Register projections in `build_user_profile_manager` (`memory_vector_wiring.py`) or custom host wiring:

```text
UserProfileManager(store, memory_projections=(...))
```

Coordinator is created inside manager when projections are non-empty.

## New Entity indexer

Implement `EntityMemoryIndexer`:

- `index_memory_entry(identity, scope, entry)`
- `remove_memory_entry(identity, scope, memory_entry_id)`

**No concrete store assumptions in callers** — inject store behind indexer.

Preserve `RequestIdentity` on every call; respect `EntityTemporalMemoryCapability` governance boundary for reads.

## Failure and recovery

If `upsert_memory_entry` raises:

- Canonical mutation may already be committed
- Lifecycle disposition → partial
- Operator/user path: `MemoryControlPlane.reconcile` or `UserProfileManager.reconcile_memory_projections`

## Extension checklist

| Question | Answer |
| -------- | ------ |
| Contract? | `UserProfileMemoryProjection` |
| Canonical truth? | UserProfile store |
| This layer? | Derived only |
| Identity? | From context only |
| Reconcile? | Required |
| Tests? | Lifecycle + reconcile unit tests |

## DO NOT

- Treat projection store as source of truth for user facts
- Write canonical UserProfile from projection callback
- Use hidden identity channels (ContextVar, globals)
