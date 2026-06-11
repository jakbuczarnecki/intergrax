---
id: IJ-2026-06-10-029
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-2
status: completed
commit: pending
adr: none — extends existing §32.8 / §40.2 ledger contract without new semantics
---

# ACP-PROD-2 declarative tool executor wired to HarnessKernel

## Operator request

Continue the ACP production sprint sequence by wiring `SideEffectLedger` to actual declarative tool execution in the step kernel, so resume replays skip committed mutating tools instead of only validating them.

## Summary

Added `execute_declarative_actions()` and `DeclarativeToolInvoker` in `intergrax/agents/persistence/declarative_tool_executor.py`. `HarnessKernel` now executes validated `requested_actions` in `DECLARATIVE` mode, emits `TOOL_COMPLETED` / `TOOL_FAILED` / `TOOL_DENIED` events, commits successful invokes to the ledger, and records execution diagnostics on the step trace. Replay-skipped keys never invoke the tool invoker.

## Project impact

Closes the architecture gap where §32.8 required harness execution after `on_next_step` but the kernel only validated and traced declarative actions. Mutating-agent resume can now dedupe side effects when a host supplies `declarative_tool_invoker` on `StepKernelContext`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §32.8, §40.2 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 7.2 ACP-PROD-2 |
| ADR | none — behavior matches existing canon |

## Changed artifacts

- `intergrax/agents/persistence/declarative_tool_executor.py` — executor + invoker protocol
- `intergrax/runtime/kernel/step_kernel.py` — execute/commit path + trace events
- `intergrax/agents/persistence/__init__.py` — exports
- `tests/unit/agents/persistence/test_declarative_tool_executor.py` — unit tests
- `tests/unit/runtime/kernel/test_step_kernel.py` — kernel integration tests
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — Wave 7.2 acceptance note

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_declarative_tool_executor.py tests/unit/runtime/kernel/test_step_kernel.py -q
uv run pytest -m gate -q
```

## Risks and follow-ups

- Hosts must supply `declarative_tool_invoker` (catalog gateway adapter) for real tool I/O; without it actions remain `skipped_no_invoker`.
- ACP-PROD-3 compensation enqueue on step failure after commit remains follow-up.
