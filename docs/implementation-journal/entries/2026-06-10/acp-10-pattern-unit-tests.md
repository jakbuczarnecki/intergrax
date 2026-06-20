---
id: IJ-2026-06-10-025
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-10
status: completed
commit: pending
adr: none — test coverage only; no contract change
---

# ACP-10 cognitive pattern unit test package

## Operator request

Continue agent-architecture sprints sequentially per plan and Cursor rules; implement the next backlog item after ACP-7.

## Summary

Expanded `tests/unit/agents/authoring/patterns/` into a structured gate package: contract/registry validation per pattern class, parametrized typed `run()` smoke for all five harness probes (mock LLM stub, no network), and phase-machine unit tests for plan-execute and reflection. Retained `test_cognitive_patterns.py` as package entry asserting full `PATTERN_AGENT_BY_ID` coverage.

## Project impact

Wave 5 pattern library is now verifiable in CI without Nexus or external LLM; each cognitive pattern has an isolated unit-test path before agent_os acceptance (ACP-12).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §21–§26 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-10 |

## Changed artifacts

- `tests/unit/agents/authoring/patterns/conftest.py` — shared `AgentRunRequest` fixture
- `tests/unit/agents/authoring/patterns/test_pattern_contracts.py` — registry + assembly validation
- `tests/unit/agents/authoring/patterns/test_pattern_probe_runs.py` — five probe typed runs
- `tests/unit/agents/authoring/patterns/test_pattern_phase_machines.py` — phase transitions
- `tests/unit/agents/authoring/patterns/test_cognitive_patterns.py` — package coverage entry
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-10 Done

## Verification

```bash
uv run pytest tests/unit/agents/authoring/patterns/ -q
```

Result: pass (32 tests).

## Risks and follow-ups

- ACP-12 agent_os acceptance (one test per pattern at acceptance tier) remains Planned.
- ReAct budget integration with TOOL-ENG-6 (DEBT-ACP-18) not covered here.
