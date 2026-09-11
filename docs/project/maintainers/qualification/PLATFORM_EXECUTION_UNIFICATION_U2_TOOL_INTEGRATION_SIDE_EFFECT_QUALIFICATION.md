# Platform Execution Unification — U2 Tool / Integration Side-Effect Closure

**Status:** `U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFIED`  
**Architecture:** [`../architecture/PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md`](../architecture/PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md)  
**P0 inventory:** [`PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md`](PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md)

## Before (BY-02 / EP-16)

```text
drain_pending_compensation_jobs
  → claim job
  → DeclarativeToolInvoker.invoke
  → external tool side effect
```

## After (canonical)

```text
drain_pending_compensation_jobs
  → claim job
  → CompensationSideEffectExecutionPort.execute
  → Execution facade / ExecutionRuntime
  → identity + authority + decision lifecycle (+ optional lineage)
  → CompensationToolInvokeSession (bound declarative invoker)
  → RuntimeToolInvoker
  → compensation side effect
```

## Reused owners

| Owner | Role in U2 |
| --- | --- |
| `ExecutionRuntime` / `Execution` facade | Root admission for each compensation side effect |
| `ExecutionBoundary` | Active execution identity + authority binding |
| `CanonicalDecisionLifecycleHost` | Governance / decision lifecycle on admitted work |
| `RuntimeToolInvoker` | Canonical tool side-effect path (via catalog invoker session) |
| `CompensationQueueStore` | Claim / complete / idempotency unchanged |

## Tests

- `tests/unit/runtime/execution/test_compensation_side_effect_admission.py`
- `tests/unit/runtime/architecture/test_platform_execution_unification_u2_tool_integration_side_effect_closure.py`
- Updated compensation queue / PCM tests under `tests/unit/agents/persistence/`

## EP-14 / EP-22

| ID | U2 verdict |
| --- | --- |
| EP-14 | **QUALIFIED / NO CHANGE** — declarative wiring already routes through `RuntimeToolInvoker` with side-effect authorization; optional `agent_runtime_governance` remains U3 |
| EP-22 | **QUALIFIED / NO CHANGE** — integrations enter only as tool provider dependencies; no production direct-mutation path proven |

## Remaining gaps

- BY-01 / EP-15 (U4)
- EP-13 optional agent governance defaults (U3)
- EP-17 ambiguous work-stage port (owner decision)
