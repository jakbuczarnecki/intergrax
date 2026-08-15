---
id: IJ-2026-06-10-024
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-7
  - ACP-DX-6
  - DEBT-ACP-04
status: completed
commit: pending
adr: none — extends existing StepOutcome canon; no new runtime semantics
---

# ACP-7 typed decision helpers and UAEP deprecation bridge

## Operator request

Continue implementing the agent architecture sprint-by-sprint per plan and Cursor rules; deliver the next backlog item after Waves 6–7.

## Summary

Extended `intergrax/agents/authoring/decisions.py` with primary author helpers (`finish`, `continue_with`, `fail_step`, `pause_for_human`, `request_replan`, `delegate_handoff`) that delegate to `StepOutcome` factories §32.0.4. Legacy UAEP `complete` / `continue_to` / `delegate_to` now emit `DeprecationWarning` and map correctly (`delegate_to` uses `MODIFY_PLAN` + `AgentHandoff`; `continue_to` stores `next_step_id` in payload). Added `to_step_outcome` re-export of the UAEP bridge. Enriched `agent_decision_to_step_outcome` for handoff and `next_step_id` diagnostics.

## Project impact

Authors have a single discoverable module for readable control-flow vocabulary on the typed loop; UAEP `decide_after_step` remains bridged without encouraging new `AgentDecision` usage.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §24.3.1 · §32.0.4 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-7 |
| ADR | none — no ADR needed |

## Changed artifacts

- `intergrax/agents/authoring/decisions.py` — primary + legacy helpers
- `intergrax/agents/authoring/__init__.py` — export surface
- `intergrax/agents/authoring/uaep_step_bridge.py` — handoff / next_step_id mapping
- `tests/unit/agents/authoring/test_decisions.py` — gate unit tests
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-7 Done
- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — §24.3.1 helpers

## Verification

```bash
uv run pytest tests/unit/agents/authoring/test_decisions.py tests/unit/agents/authoring/test_uaep_step_bridge.py -q
```

Result: pass (11 tests).

## Risks and follow-ups

- ACP-10 pattern unit package depth and ACP-12 agent_os acceptance remain Planned.
- Legacy `decide_after_step` on `IntergraxAgent` still uses deprecated `complete()` internally — migrate in DEBT-ACP-05 cleanup sprint.
