# NPSC-5E — Recovery, Checkpoint & Retry Architecture

> **Stage:** P0 inventory + P0A lineage reconciliation (no R1/R2/R3 implementation)

## P0 inventory

NPSC-5E introduces a recovery plane on top of the frozen NPSC-5D governance baseline. P0 establishes boundaries before retry (R1), checkpoint lineage hardening (R2), and partial recovery (R3).

| Plane | Current owner | Lineage role |
|---|---|---|
| Execution lifecycle | `ExecutionRuntime` | records admission facts only |
| Attempt lifecycle | `AttemptLifecycleService` | per-attempt scope; no attempt mint |
| Checkpoint / resume | `LongRunningCoordinator` | segment close/open continuity |
| Orchestration | Nexus | propagate identity; optional seal on retry |
| Terminal | `ExecutionTerminalService` | lineage seal maps outcome; does not terminate |
| Governance / policy | governance plane | lineage may store provenance snapshots |

## P0A lineage reconciliation

Qualified baseline: `a72c9b568c61e28180756059ae48a99fb56eaa19` (post `a3e719b1c` lineage admission + `a72c9b568` durability hardening).

Hard invariants certified in P0A:

- Lineage ≠ lifecycle
- Lineage ≠ authority (no mint/widen)
- Lineage ≠ policy
- Lineage ≠ checkpoint (`ExecutionLineageRecord` ≠ `RuntimeCheckpoint`)
- Lineage ≠ retry decision
- Lineage ≠ scheduler
- No duplicate execution registry / attempt store / checkpoint store

See: `docs/project/maintainers/qualification/NPSC_5E_P0A_EXECUTION_LINEAGE_BASELINE_QUALIFICATION.md`

## Ownership (canonical)

```text
ExecutionRuntime           = lifecycle root
AttemptLifecycleService    = attempt transition authority
ExecutionLineagePersistence = provenance persistence
Nexus                      = orchestration / scheduling
LongRunningCoordinator     = checkpoint/resume orchestration
Governance                 = WHETHER
Execution authority        = canonical authority owner
```

## Baseline

```text
NPSC-5D FROZEN: a4a1faca01cd5004e372f235132184a84aa5a6bd
NPSC-5E LINEAGE BASELINE: a72c9b568c61e28180756059ae48a99fb56eaa19
```

## Future boundaries

### R1 — Canonical Execution Retry & Attempt Semantics

- `AttemptLifecycleService.transition_to_next_attempt` remains sole retry attempt mint path.
- Lineage represents attempt1 → attempt2 provenance; does not decide RETRY/FAIL/RESUME.

### R2 — Checkpoint Lineage Hardening

- Validate resumed execution/run/attempt lineage against checkpoint facts (gap recorded in P0A).
- Checkpoint remains source of truth for restorable state.

### R3 — Partial Recovery / Fan-out Continuation

- Preserve NPSC-5B two-level fan-out lineage.
- Exact slot resume must not create false sibling lineage.
- Governed HITL continuation ≠ failure retry.

## Deferred / out of scope (P0A)

- `RecoveryLineageManager`, `RetryLineageEngine`, `ExecutionLineageRuntime` — forbidden
- KV lineage adapter
- Wrong-checkpoint lineage validation (R2)
- New schema version gate beyond codec v1 fail-closed
