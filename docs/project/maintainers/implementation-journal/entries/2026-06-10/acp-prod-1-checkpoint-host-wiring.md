---
id: IJ-2026-06-10-028
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-1
  - DEBT-ACP-17
status: completed
commit: pending
adr: none — wires existing ACP-PROD-1 store into hosts and Nexus
---

# ACP-PROD-1 checkpoint host wiring and resume acceptance

## Operator request

Continue sequential agent-architecture sprints; deliver production checkpoint depth after scaffold default typed (DEBT-ACP-05).

## Summary

Added `checkpoint_wiring.py` (`wire_acp_run_request`, `open_agent_checkpoint_store`, Nexus metadata injection). `GraphExecutor` and `NexusLoop` accept `agent_checkpoint_store`; lab host wires SQLite `agent_checkpoints.db` + task enricher. `runtime_request_to_agent_run` bridges `user_id` and `acp.execution_options.v1`. Acceptance `test_acceptance_05c_acp_checkpoint_resume` proves crash-after-one-step resume on PlanExecute probe without replaying all perceives.

## Project impact

Typed ACP sessions can persist and resume step state when hosts inject `AgentCheckpointStore`; lab application provides default wiring for experiments.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.1 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-PROD-1 · Wave 7.1 |

## Changed artifacts

- `intergrax/agents/persistence/checkpoint_wiring.py`
- `intergrax/applications/_shared/acp_checkpoint_task_enricher.py`
- `intergrax/runtime/nexus/execution/graph_executor.py`
- `intergrax/runtime/nexus/nexus_loop.py`
- `intergrax/applications/_shared/nexus_factory.py`, `harness_host_runtime.py`
- `applications/lab_application/host/integration_wiring.py`, `factory.py`
- `intergrax/agents/runtime_request_bridge.py`
- `tests/unit/agents/persistence/test_acp_checkpoint_resume.py`, `test_checkpoint_wiring.py`
- `tests/acceptance/agent_os/test_acp_checkpoint_resume.py`

## Verification

```bash
uv run pytest tests/unit/agents/persistence/ tests/acceptance/agent_os/test_acp_checkpoint_resume.py -q
```

Result: pass (10 tests).

## Risks and follow-ups

- ACP-PROD-2 idempotency replay on real tool invoke remains shallow.
- ACP-PROD-3 compensation enqueue not implemented.
