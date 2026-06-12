---
id: IJ-2026-06-11-029
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-TOK-2
status: completed
commit: pending
adr: none — enforcement on existing ResolvedBudgetLimits + StepLLMRouter wrapper
---

# Sprint 4 ACP-TOK-2 — hard/advisory token budget enforcement

## Operator request

Continue ACP-FINISH sprint queue with ACP-TOK-2: enforce per-agent and environment token limits with hard block before LLM and advisory metering-only mode.

## Summary

Added `evaluate_hard_budget_violation` and `BudgetEnforcingLLMRouter` wrapper. `AgentRuntime.advance_step` checks step boundaries; `HarnessKernel` handles `BUDGET_EXCEEDED` outcomes without policy misclassification. Hard caps block the next LLM call; advisory limits expose `tokens_remaining` without blocking. Request-level `max_total_tokens` defaults to HARD enforcement.

## Project impact

Hosts using `AgentBinding.budget_slice` with `enforcement=hard` now get kernel-enforced token caps on direct `agent.run()` sessions. Closes GAP-ACP-37 implementation path for hard enforcement; reaction policies remain ACP-TOK-3.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §25.5.1–§25.5.2 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-TOK-2 |
| Cross-plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` §43 kernel hard cap row |

## Changed artifacts

- `intergrax/contracts/acp_budget_enforcement.py` — violation evaluation
- `intergrax/agents/acp_budget_enforcement_bridge.py` — outcome + router error types
- `intergrax/agents/authoring/budget_enforcing_llm_router.py` — pre-LLM wrapper
- `intergrax/agents/authoring/step_loop.py` — step boundary enforcement
- `intergrax/runtime/kernel/step_kernel.py` — budget outcome handling + `check_hard_budget_before_llm`
- `intergrax/agents/authoring/acp_run.py` — wire enforcing router
- `tests/unit/agents/test_acp_token_budget_enforcement.py`

## Verification

```bash
uv run pytest tests/unit/agents/test_acp_token_budget_enforcement.py tests/unit/agents/test_acp_token_usage_metering.py tests/unit/runtime/kernel/test_step_kernel.py -q
```

Result: pass (19 tests).

## Risks and follow-ups

- ACP-TOK-3: `BUDGET_THRESHOLD` / `BUDGET_EXCEEDED` events and `BudgetReactionProfile` reactions (HITL, notify, degrade_model).
- APP-PROD-7 gate can now be wired against hosts with `budget_slice` + HARD enforcement.
