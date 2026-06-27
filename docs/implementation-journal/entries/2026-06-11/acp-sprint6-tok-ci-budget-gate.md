---
id: IJ-2026-06-11-031
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-TOK-CI
status: completed
commit: pending
adr: none — CI enforcement of existing §25.4–§25.5 contracts
---

# Sprint 6 ACP-TOK-CI — token budget contract CI gate

## Operator request

Continue ACP-FINISH sprint queue with ACP-TOK-CI: add CI gate that fails when kernel bypasses token metering or Tier-2 agents mutate budget counters via `state_delta`.

## Summary

Added `scripts/maintenance/check_agent_token_budget_contract.py` (CI-18). Static checks verify `HarnessKernel` calls `apply_llm_metering_after_step`, `acp_run` wraps the enforcing LLM router, and `step_loop` uses `handle_hard_budget_violation`. Agents are scanned for forbidden metering imports and budget keys in `state_delta`. Smoke step runs the three ACP-TOK unit test modules. Wired into `check_acp_ci_conformance_matrix.py` and `check_agent_acp_close_ci.py`.

## Project impact

Token budget contract is now CI-enforced — authors cannot bypass harness-owned metering or self-increment budget counters. Completes ACP-FINISH runtime depth track; only ACP-FINISH-DOC-1 remains for architecture doc truth.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.10 CI-18 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-TOK-CI |
| Cross-plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` §43 CI gate row |

## Changed artifacts

- `scripts/maintenance/check_agent_token_budget_contract.py` — CI-18 gate (new)
- `scripts/gates/check_acp_ci_conformance_matrix.py` — CI-18 row
- `scripts/gates/check_agent_acp_close_ci.py` — aggregate includes TOK-CI
- `tests/unit/scripts/test_check_agent_token_budget_contract.py`

## Verification

```bash
uv run python scripts/maintenance/check_agent_token_budget_contract.py
uv run pytest tests/unit/scripts/test_check_agent_token_budget_contract.py -m gate -q
```

Result: pass.

## Risks and follow-ups

- ACP-FINISH-DOC-1: mark GAP-ACP-36/37 Closed in architecture §28.3 and refresh §40.13.
- APP-PROD-7: Tier-3 host production gate for `budget_slice` + HARD enforcement.
