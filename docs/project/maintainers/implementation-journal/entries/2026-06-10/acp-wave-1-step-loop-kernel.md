---
id: IJ-2026-06-10-010
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-STEP-1
  - ACP-STEP-2
  - ACP-STEP-2b
  - ACP-CON-3
status: completed
commit: pending
adr: none — implements accepted §32 · §38 split; no semantic change to ADR-AGENT-001..003
---

# ACP Wave 1 — step loop glue and HarnessKernel

## Operator request

Implement the next ACP sprint: `on_next_step` author hook, glue-only `AgentRuntime.advance_step`, and `HarnessKernel.execute_step` with policy, state merge, budgets, and trace records.

## Summary

Delivered the agent iteration stack: expanded `AgentStepContext`, default `IntergraxAgent.on_next_step` (with `@step` bridge via `uaep_exec_ctx` metadata), `AgentRuntime.advance_step` delegating exclusively to `HarnessKernel`, L1 kernel cycle in `intergrax/runtime/kernel/step_kernel.py`, and `ACP-CON-3` side-effect mode validation.

## Project impact

One harness iteration is now expressible as **domain decides → kernel executes** with testable invariants. Wave 2 `agent.run(AgentRunRequest)` can wrap a multi-step loop around `advance_step` without duplicating policy or merge logic.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §32 · §38 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 1 |
| ADR | ADR-AGENT-003 |

## Changed artifacts

- `intergrax/agents/authoring/step_loop.py` — `AgentRuntime.advance_step`
- `intergrax/agents/authoring/base.py` — default `on_next_step`
- `intergrax/runtime/kernel/step_kernel.py` — `HarnessKernel.execute_step`
- `intergrax/agents/authoring/side_effect_validation.py` — ACP-CON-3
- `intergrax/contracts/step_execution.py` — `StepExecutionRecord`, `AgentStepRecord`
- `intergrax/contracts/agent_step_context.py` — expanded context fields
- `tests/unit/runtime/kernel/` · `tests/unit/agents/authoring/test_step_loop.py`

## Verification

```bash
uv run pytest tests/unit/agents/authoring/ tests/unit/runtime/kernel/ -m gate -q
uv run pytest tests/unit/runtime/registry/ tests/unit/agents/ -m gate -q
```

9 Wave 1 tests + 42 registry/agents gate tests green.

## Risks and follow-ups

- Declarative `requested_actions` execution is trace-only stub until TOOL-ENG-6 sync (Wave 5 / tools plan).
- `AgentStepRecord` on `AgentRunTrace` uses dict wire form until ACP-OBS-1 typed step records.
- Next: Wave 2 — `merge_environment`, `IntergraxAgent.run(AgentRunRequest)`, Nexus bridge.
