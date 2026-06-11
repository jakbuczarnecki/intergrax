---
id: IJ-2026-06-11-008
date: 2026-06-11
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-PROD-5
status: completed
commit: pending
adr: none — durable queue for existing CompensationRequest contract
---

# ACP-CLOSE PROD-5 — durable compensation queue

## Operator request

Execute next ACP-CLOSE sprint: persist `enqueued` compensation requests when declarative invoker is unavailable at step-failure time.

## Summary

Added `CompensationQueueStore` (in-memory + SQLite), `compensation_queue_wiring`, and `drain_pending_compensation_jobs` worker. `enqueue_compensations_for_step_failure` persists jobs when `compensation_queue` is set and invoker is absent. Wired through `StepKernelContext`, `acp_run`, Nexus `graph_executor`, harness host runtime, and task enricher alongside agent checkpoints.

## Project impact

Policy-deny-after-commit paths can defer compensation tool invokes to a durable host queue instead of losing `enqueued` status. GAP-ACP-27 fully closed.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.3.3 · GAP-ACP-27 |
| Plan | `ACP-CLOSE-PROD-5` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/persistence/compensation_queue_store.py` — store port (new)
- `intergrax/agents/persistence/compensation_queue_wiring.py` — host wiring (new)
- `intergrax/agents/persistence/compensation_queue_worker.py` — drain worker (new)
- `intergrax/agents/persistence/compensation_enqueue.py` — persist enqueued jobs
- `intergrax/runtime/kernel/step_kernel.py` — queue on kernel context
- `intergrax/agents/authoring/acp_run.py` — resolve queue from metadata
- `intergrax/applications/_shared/acp_checkpoint_host_wiring.py` — host store resolution
- `intergrax/applications/_shared/harness_host_runtime.py` — expose store
- `intergrax/runtime/nexus/nexus_loop.py`, `graph_executor.py` — Nexus inject
- `tests/unit/agents/persistence/test_compensation_queue_*.py` — unit tests (new)

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_compensation_enqueue.py tests/unit/agents/persistence/test_compensation_queue_store.py tests/unit/agents/persistence/test_compensation_queue_worker.py -q
```

## Risks and follow-ups

- Background scheduler to call `drain_pending_compensation_jobs` on product hosts not wired this sprint.
- ACP-CLOSE-PROD-6 idempotency store cross-run depth remains open.
