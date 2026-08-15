---
id: IJ-2026-06-10-022
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-4
  - ACP-PROD-5
status: completed
commit: pending
adr: none — implements architecture §40.4–§40.5
---

# Reliability profile in kernel and shared context CAS

## Operator request

Execute the next ACP sprint: Wave 7 reliability and shared-context concurrency (ACP-PROD-4, ACP-PROD-5).

## Summary

Wired `ReliabilityProfile` into `AgentSessionReliability` on the harness kernel (circuit breaker, checkpoint interval). Extended `SharedContextView` with per-key `publish`, `compare_and_swap`, conflict policies, and bridged entry versions through Nexus `SharedTaskContext`.

## Project impact

Agent sessions inherit host reliability posture; parallel graph nodes can coordinate via optimistic locking on shared keys with measurable conflict detection.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.4–§40.5 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 7 steps 7.4–7.5 |
| Cross-domain | `RELIABILITY_FAILURE_AND_HITL`, `ORCHESTRATION` |

## Changed artifacts

- `intergrax/runtime/kernel/session_reliability.py`, `step_kernel.py`
- `intergrax/contracts/shared_context.py`
- `intergrax/runtime/nexus/context/shared_task_context.py`
- `intergrax/agents/authoring/shared_context_bridge.py`, `acp_run.py`
- `tests/unit/runtime/kernel/test_session_reliability_acp_prod4.py`
- `tests/unit/agents/authoring/test_shared_context_view.py`

## Verification

```bash
uv run pytest tests/unit/runtime/kernel/test_session_reliability_acp_prod4.py tests/unit/agents/authoring/test_shared_context_view.py tests/unit/runtime/kernel/test_step_kernel.py -q -m gate
```

Result: pass.

## Risks and follow-ups

- Tool-level retry/backoff not yet executed in kernel — circuit breaker only.
- HITL-on-conflict policy raises error; Nexus HITL wiring deferred.
- ACP-PROD-6..11 remain open.
