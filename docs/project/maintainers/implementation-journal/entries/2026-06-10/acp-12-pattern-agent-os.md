---
id: IJ-2026-06-10-026
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-12
status: completed
commit: pending
adr: none — acceptance tests only; exercises existing Nexus→ACP bridge
---

# ACP-12 cognitive pattern agent_os acceptance

## Operator request

Continue sequential agent-architecture sprints; implement the next planned item after ACP-10.

## Summary

Added `tests/acceptance/agent_os/test_acp_pattern_agents.py` with five parametrized acceptance tests — one per cognitive pattern harness probe. Each runs `Task → NexusLoop → AgentEngine` with `acp.session.v1` enabled (typed ACP session), mock LLM inside reference probes, and asserts `TaskState.COMPLETED` with non-empty answer.

## Project impact

Wave 5 pattern library is demonstrated end-to-end through Tier-1 Agent OS routing, not only direct `AgentRunRequest` unit runs.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §21–§26 · §35 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-12 |

## Changed artifacts

- `tests/acceptance/agent_os/test_acp_pattern_agents.py` — five pattern acceptance tests
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-12 Done; removed stale duplicate ACP-13 Planned row

## Verification

```bash
uv run pytest tests/acceptance/agent_os/test_acp_pattern_agents.py -q
```

Result: pass (5 tests).

## Risks and follow-ups

- `runtime_request_to_agent_run` should fall back to `RuntimeRequest.user_id` when metadata omits `user_id` — tests pass `user_id` in metadata explicitly.
- Next backlog: DEBT-ACP-05 legacy scaffold cleanup or PROD hardening sprints.
