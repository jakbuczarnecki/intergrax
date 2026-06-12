---
id: IJ-2026-06-11-028
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-TOK-1
status: completed
commit: pending
adr: none — extends existing AcpBudgetState / HarnessKernel metering contract
---

# Sprint 3 ACP-TOK-1 — invocation token metering rollups

## Operator request

Continue the post-freeze sprint backlog with ACP-TOK-1: populate agent and environment token rollups in invocation state so authors can read `invocation_usage` and `budget.tokens_*` during `on_next_step`.

## Summary

Implemented token metering in `HarnessKernel` after LLM call drain: `acp.state.v1.budget` counters increment from `LlmCallRecord`, environment rollup persists under `acp.usage.v1` in step metadata, and `step_ctx.invocation_usage` refreshes before the next step. `merge_environment` now materializes `ResolvedBudgetLimits` (limits + `tokens_remaining` without enforcement). Added bridge module, budget resolution, and integration tests.

## Project impact

ACP §25.4 metering is live — authors can implement adaptive `model_hint` and soft budget strategies from `invocation_usage`. Closes GAP-ACP-36 implementation path; hard enforcement remains ACP-TOK-2.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §25.4 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-TOK-1 |
| Cross-plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` §43 token metering row |

## Changed artifacts

- `intergrax/contracts/acp_token_metering.py` — pure metering helpers
- `intergrax/agents/acp_budget_resolution.py` — limit merge for `merge_environment`
- `intergrax/agents/acp_token_metering_bridge.py` — kernel/session bridge
- `intergrax/agents/run_environment.py` — `resolved_budget_limits` on merged env
- `intergrax/runtime/kernel/step_kernel.py` — post-step metering hook
- `intergrax/agents/authoring/acp_run.py` — seed limits + persist `acp.usage.v1`
- `tests/unit/agents/test_acp_token_usage_metering.py` — acceptance tests

## Verification

```bash
uv run pytest tests/unit/agents/test_acp_token_usage_metering.py tests/unit/runtime/kernel/test_step_kernel.py -q
```

Result: pass.

## Risks and follow-ups

- ACP-TOK-2 hard pre-LLM enforcement and ACP-TOK-3 reaction policies remain open.
- APP-CON-3 `ActiveBudgetState` live totals await wiring from `acp.usage.v1` (optional cross-sync).
