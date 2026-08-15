---
id: IJ-2026-06-11-030
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-TOK-3
status: completed
commit: pending
adr: none — reactions on existing BudgetReactionProfile + RuntimeEvent spine
---

# Sprint 5 ACP-TOK-3 — budget reaction policies and runtime events

## Operator request

Continue ACP-FINISH sprint queue with ACP-TOK-3: wire `BudgetReactionProfile` reactions, emit `BUDGET_THRESHOLD` / `BUDGET_EXCEEDED` runtime events, and test abort, HITL, degrade_model, notify_only, and custom_hook paths.

## Summary

Added `acp_budget_reactions.py` with `handle_hard_budget_violation` and `maybe_emit_budget_threshold`. Extended `StepKernelContext` with reaction profile, notification adapter, custom hook, and degrade flag. `HarnessKernel` emits threshold events after metering; `step_loop` delegates exceed handling to reaction policy. `BudgetEnforcingLLMRouter` forces cheapest model when degrade is active. `acp_run.py` skips output validation for `PAUSED` (HITL budget pause).

## Project impact

Application hosts can configure `CostProfile.budget_reaction` with notify channels and custom hooks; hard cap exceed triggers HITL pause, notify-only failure, model downgrade, or host callback instead of generic abort only. Closes GAP-ACP-37 reaction path; enables APP-PROD-7 host gate wiring.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §25.5.3 · §30.8 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-TOK-3 |
| Cross-plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` §43 host notify row |

## Changed artifacts

- `intergrax/runtime/events/runtime_event.py` — `BUDGET_THRESHOLD`, `BUDGET_EXCEEDED`
- `intergrax/contracts/budget_reaction_hook.py` — `BudgetReactionHook` protocol
- `intergrax/agents/acp_budget_reactions.py` — reaction dispatch + threshold emit
- `intergrax/runtime/kernel/step_kernel.py` — threshold hook + public `emit_runtime_event`
- `intergrax/agents/authoring/step_loop.py` — `handle_hard_budget_violation`
- `intergrax/agents/authoring/budget_enforcing_llm_router.py` — degrade model selection
- `intergrax/agents/authoring/acp_run.py` — host wiring + PAUSED validation skip
- `intergrax/agents/authoring/acp_session_host.py` — notify + hook fields
- `tests/unit/agents/test_acp_token_budget_reactions.py`

## Verification

```bash
uv run pytest tests/unit/agents/test_acp_token_budget_reactions.py tests/unit/agents/test_acp_token_budget_enforcement.py tests/unit/runtime/kernel/test_step_kernel.py -q
```

Result: pass (20 tests).

## Risks and follow-ups

- ACP-TOK-CI: static + smoke gate for kernel metering contract.
- ACP-FINISH-DOC-1: close GAP-ACP-36/37 register rows in architecture §28.3.
