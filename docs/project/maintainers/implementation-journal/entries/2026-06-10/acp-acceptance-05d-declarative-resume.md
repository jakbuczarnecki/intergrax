---
id: IJ-2026-06-10-032
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-1
  - ACP-PROD-2
status: completed
commit: pending
adr: none — acceptance coverage for existing persistence contracts
---

# ACP acceptance 05d — declarative mutating tool resume without double invoke

## Operator request

Continue the ACP production sprint sequence with an agent_os acceptance test proving checkpoint resume skips replay of committed mutating declarative tools.

## Summary

Added `test_acceptance_05d_acp_declarative_mutating_resume` with a harness probe agent that emits declarative `requested_actions` in `DECLARATIVE` mode, checkpoints after the first tool commit, and on resume re-issues the same `idempotency_key` without a second invoke.

## Project impact

Closes Wave 7 DoD evidence for “no double mutating tool” on the typed ACP path (ledger + invoker + checkpoint), complementing `test_acceptance_05c`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.1–§40.2 |
| Plan | Wave 7.1 ACP-PROD-1 · agent_os scenario 5 |

## Changed artifacts

- `tests/acceptance/agent_os/test_acp_declarative_mutating_resume.py`
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — agent_os scenario table

## Verification

```bash
uv run pytest tests/acceptance/agent_os/test_acp_declarative_mutating_resume.py -q
```

## Risks and follow-ups

- Probe uses in-memory invoker; Nexus graph path with host catalog invoker is covered separately by unit wiring tests.
