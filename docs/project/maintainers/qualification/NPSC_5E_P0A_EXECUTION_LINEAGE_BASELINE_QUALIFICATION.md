# NPSC-5E-P0A — Execution Lineage Baseline Qualification

> **Task:** `NPSC-5E-P0A`  
> **Architecture:** `docs/project/maintainers/architecture/NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md`  
> **Predecessor freeze:** `a4a1faca01cd5004e372f235132184a84aa5a6bd` (NPSC-5D Final)  
> **Session START_HEAD / START_ORIGIN:** `0e722f285b178b8c456bb6d9e49468c07765cf82`  
> **Qualified production baseline (pre-P0A commit):** `a72c9b568c61e28180756059ae48a99fb56eaa19`

## Purpose

Formal cross-layer qualification proving `ExecutionLineage*` is canonical provenance persistence only — not a second runtime, lifecycle engine, attempt authority, scheduler, checkpoint store, or policy/authority mint.

## Drift range

Comparison: `a4a1faca01cd5004e372f235132184a84aa5a6bd..a72c9b568c61e28180756059ae48a99fb56eaa19`

| Class | Commits / paths |
|---|---|
| **A — Execution / Lineage / Nexus (relevant)** | `a3e719b1c` — durable execution lineage admission persistence; `a72c9b568` — lineage durability hardening + host/nexus wiring; lineage surface under `intergrax/contracts/execution_lineage.py`, `intergrax/runtime/execution/lineage/`, wiring in `runtime.py`, `child.py`, `host_task.py`, `nexus_loop.py`, `graph_runner.py`, `long_running_bridge.py`, `nexus_factory.py` |
| **B — unrelated VPI** | `756f36484`, `0e722f285` — lexical retrieval |
| **C — unrelated platform scenario** | VPI PostgreSQL adapters, integration tests |
| **D — unrelated Decision** | `c6df1d975` — revision context alignment projection |
| **E — docs/tests only** | `DG_001_EXECUTION_LINEAGE_ADMISSION_PERSISTENCE_R1.md`, lineage unit tests |

## Execution lineage architecture

`ExecutionLineagePersistence` records durable admission facts for root/child executions within an `ExecutionLineageAttemptScope` (tenant, task, run, attempt). Active propagation uses execution-scoped `ContextVar` carriers (`ActiveExecutionLineageState`). Root activation opens attempt + segment before delegate; admission hooks run before `ExecutionBoundary` delegate.

## Ownership matrix

| Concern | Owner |
|---|---|
| Execution lifecycle | `ExecutionRuntime` |
| Execution identity mint | `identity_authority.py` via `ExecutionRuntime` / `ExecutionBoundary` |
| Run / attempt identity | `ExecutionRuntime` + `AttemptLifecycleService` |
| Attempt transition / retry | `AttemptLifecycleService.transition_to_next_attempt` |
| Lineage persistence | `ExecutionLineagePersistence` implementations |
| Lineage admission | `ExecutionLineageRootAdmissionHook` / `ExecutionLineageChildAdmissionHook` |
| Root lineage activation | `activate_root_execution_lineage` |
| Child lineage propagation | `ChildExecutionRunner` + active lineage peek |
| Nexus lineage propagation | optional `execution_lineage_persistence` on graph runner / long-running bridge |
| Long-running lineage restore | `segment_predecessor_root_execution_id` on `ExecutionRuntime` + `close_active_lineage_segment_for_resume` |
| Checkpoint | `LongRunningCoordinator` / `TaskCheckpointPersistence` |
| Retry decision | retry policy + `AttemptLifecycleService` (not lineage) |
| Terminal execution | `ExecutionTerminalService` |
| Lineage seal | `seal_lineage_attempt` records closure; does not terminate execution |
| Governance / policy | governance plane (lineage may store provenance only) |
| Authority | canonical execution authority owners (lineage does not mint) |
| Scheduling | Nexus orchestration |

## Identity model

- Segment root = canonical root `ExecutionId` for an attempt segment.
- Child admissions reference exact parent `ExecutionId`.
- Retry: same `RunId`, new `AttemptId` via `AttemptLifecycleService`; separate lineage attempt scopes with segment predecessor chain.

## Root admission

Order (repository-defined): execution identity exists → `activate_root_execution_lineage` (open attempt/segment) → lineage root admission hook before delegate. No synthetic execution identity from lineage.

## Child propagation

`ChildExecutionRunner` peeks active lineage and prepends child admission hook. Wrong parent fails closed (`ExecutionLineageIntegrityError`).

## Attempt interaction

Lineage scopes are keyed by `attempt_id`. Retry supersede seals prior attempt (`RETRY_SUPERSEDED` from graph runner on canonical transition). Lineage does not mint attempts.

## Nexus propagation

`graph_runner` may seal lineage on retry transition; `long_running_bridge` closes segment on checkpoint. Neither mints execution identity.

## Long-running resume

Checkpoint persists task state; lineage closes segment cleanly (`close_segment_for_resume`). Successor segment references `predecessor_root_execution_id`. Unclean predecessor marks attempt degraded.

## Persistence / idempotency

Providers: `InMemoryExecutionLineagePersistence`, `DocumentStoreExecutionLineagePersistence`. `_verify_idempotent_admission` ensures duplicate identical facts are idempotent; conflicting facts fail closed. Codec schema version `1`; unknown version fails closed.

## Concurrency isolation

`ContextVar` active lineage; parallel sibling admissions use CAS with monotonic `admission_position`. Tenant partitions isolate scope.

## R3 compatibility

NPSC-5D R3 frozen regression modules remain importable. Governed continuation (pause/approval/resume) is distinct from failure retry; lineage records segment continuity only.

## Recovery readiness assessment

| Gate | Verdict |
|---|---|
| ATTEMPT LINEAGE READY FOR 5E/R1 | **YES** — per-attempt scopes + `RETRY_SUPERSEDED` seal + `AttemptLifecycleService` transition |
| CHECKPOINT LINEAGE READY FOR 5E/R2 | **YES** — segment predecessor chain; **gap:** wrong checkpoint/lineage cross-validation deferred to R2 |
| PARTIAL RECOVERY LINEAGE READY FOR 5E/R3 | **YES** — fan-out two-level lineage preserved per NPSC-5B freeze |

## Regression matrix

| Suite | Status |
|---|---|
| NPSC-5A/B/C/D frozen modules | PASS (import gate + frozen regressions) |
| `test_execution_lineage_contracts` | PASS |
| `test_execution_lineage_admission_order` | PASS |
| `test_execution_lineage_persistence_conformance` | PASS |
| Execution runtime / attempt / child focused | PASS |
| Nexus graph_runner / nexus_loop / orchestration | PASS |
| Long-running checkpoint / resume | PASS |
| R3 HITL continuation | PASS |
| P0A qualification test | PASS |
| ruff / pyright (qualification scope) | PASS |

## Formal verdict

**PASS / BASELINE QUALIFIED**

`ExecutionLineage*` is provenance-only. No duplicate runtime, scheduler, attempt authority, or checkpoint store detected.

## Baseline declaration

```text
NPSC-5D FROZEN BASE:
a4a1faca01cd5004e372f235132184a84aa5a6bd

NPSC-5E LINEAGE-ENABLED BASELINE:
a72c9b568c61e28180756059ae48a99fb56eaa19
```

Qualification commit is metadata only; production baseline is the SHA above.

## Status

```text
NPSC-5E-P0 = COMPLETE
NPSC-5E-P0A = PASS / BASELINE QUALIFIED
NPSC-5E = ACTIVE
NEXT: NPSC-5E/R1 — Canonical Execution Retry & Attempt Semantics
```
