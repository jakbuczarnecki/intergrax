---
id: IJ-2026-06-10-030
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-3
status: completed
commit: pending
adr: none — implements existing §40.3 compensation contract
---

# ACP-PROD-3 compensation enqueue on step failure after commit

## Operator request

Continue the ACP production sprint sequence with compensation enqueue when a step fails after mutating tools were committed, completing the §40.3 follow-up deferred from the initial ACP-PROD-3 profile gate.

## Summary

Added `CompensationRequest` to `intergrax/contracts/side_effect.py` and `enqueue_compensations_for_step_failure()` in `compensation_enqueue.py`. `SideEffectLedger` now exposes `committed_for_step`, `mark_failed`, and `mark_compensated`. `HarnessKernel` invokes compensation on `policy_post` deny when committed effects exist in the current step, emits compensation trace events, and accumulates requests on `StepKernelContext.compensation_requests`.

## Project impact

Mutating declarative agents can roll back committed side effects when a later gate in the same step fails (e.g. output policy deny). Manual-reversibility tools surface HITL via `HUMAN_APPROVAL_REQUESTED` instead of silent failure.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.3 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 7.3 ACP-PROD-3 |
| ADR | none |

## Changed artifacts

- `intergrax/contracts/side_effect.py` — `CompensationRequest`
- `intergrax/agents/persistence/compensation_enqueue.py` — enqueue + invoke policy
- `intergrax/agents/persistence/side_effect_ledger.py` — status transitions
- `intergrax/runtime/kernel/step_kernel.py` — policy_post failure hook
- `tests/unit/agents/persistence/test_compensation_enqueue.py`
- `tests/unit/runtime/kernel/test_step_kernel.py`

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_compensation_enqueue.py tests/unit/runtime/kernel/test_step_kernel.py -q
```

## Risks and follow-ups

- Compensation without `declarative_tool_invoker` is enqueued only (not executed).
- Async compensation worker / durable queue for enqueued requests remains host responsibility.
