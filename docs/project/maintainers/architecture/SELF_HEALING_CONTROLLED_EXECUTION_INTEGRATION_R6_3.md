# Self-Healing R6.3 — Controlled Execution Integration

## Purpose

R6.3 wires **pre-execution autonomy controls** into the existing **Execution Spine** without introducing a new executor, workflow, or lifecycle.

**Guard controls execution admission. Guard does not execute.**

## Placement

```text
Recommendation (R5.3)
        |
        v
Autonomy Control (R6.1)
        |
        v
Decision Evaluation (R6.2)
        |
        v
AutonomyExecutionGuard (R6.3)  ← admission only
        |
        v
ExecutionBoundary (existing UE-1B spine)
        |
        v
Existing Executor / Delegate
```

Integration surface:

| Layer | Artifact |
|-------|----------|
| Contracts | `AutonomyExecutionBoundary`, `AutonomyExecutionAuthorization`, `AutonomyExecutionAuditRepository` |
| Runtime | `DefaultAutonomyExecutionGuard`, `DefaultAutonomyExecutionBoundary` |
| Spine hook | `ExecutionAdmissionHook` on `ExecutionBoundary` — no change to delegate semantics |

Hosts attach autonomy via `DefaultAutonomyExecutionBoundary.spine_admission_hook(source)`. When `AutonomyAdmissionContextSource.admission_for` returns `None`, the legacy path is unchanged.

## What the guard checks

- Persisted **evaluation** exists for the recommendation correlation (fail-safe: missing → `DENIED`)
- Evaluation verdict is not `DENIED`
- Prior **control decision** permits controlled auto path (`CONTROLLED_EXECUTION`, `auto_path_allowed`)
- Human **approval token** present when evaluation requires approval
- Optional **guard rule plugins** (`AutonomyExecutionGuardRule`)

## What the guard does NOT do

- Invoke executors or strategy runners
- Mutate requests or select executors
- Replace execution authority, retry, or idempotency
- Bypass `ExecutionBoundary` / `ExecutionRuntime`

## Authorization model

`AutonomyExecutionAuthorization` (immutable):

| Status | Spine behavior |
|--------|----------------|
| `AUTHORIZED` | Admission hook allows delegate execution |
| `CONDITIONAL` | Recorded; **does not** permit execution (fail-safe) |
| `DENIED` | Admission hook raises `AutonomyExecutionDeniedError` |

Audit persistence uses port `AutonomyExecutionAuditRepository` only (in-memory adapter for tests).

## Autonomy / execution boundary

Autonomy packages **must not** import executor implementations. The only runtime coupling to execution is `DefaultAutonomyExecutionBoundary` → `ExecutionAdmissionHook` protocol in `intergrax/runtime/execution/boundary.py`.

## Compatibility

Paths without autonomy admission context continue:

```text
ExecutionBoundary.execute(request)
        |
        v
Delegate (unchanged)
```

Autonomy is an **extension**, not a replacement for governed execution.
