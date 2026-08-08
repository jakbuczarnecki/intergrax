---
id: IJ-2026-06-10-009
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-DX-1
  - ACP-CON-1
  - ACP-CON-4
  - ACP-0
  - ACP-DX-6
  - ACP-CON-2
status: completed
commit: pending
adr: none — implements accepted canon §29 · §32.0 · §37.1–2 · §12 gate per ADR-AGENT-002/003
---

# ACP Wave 0 — typed run/step/state contracts

## Operator request

Execute the first ACP implementation sprint (Wave 0): ship typed `AgentRunRequest`/`Result`, session state envelope, `StepOutcome` factories, state merge, §12 assembly gate, and CI guard for raw dict state in Tier-2 agents.

## Summary

Delivered Tier-0 contracts and authoring helpers for the agent session model: `intergrax/contracts/agent_run.py` (+ enums), `acp_state.py`, minimal `AgentStepContext`, `StepOutcome` factories, `load_session_state` / `session_state_delta`, RFC 7396 merge in `state_merge.py`, extended `validate_contract_metadata` for full §12 gate, default §12 fields on `AgentContract`, and `scripts/maintenance/check_agent_typed_state.py` wired into CI.

## Project impact

Wave 1 (`on_next_step` + `HarnessKernel.execute_step`) can now bind to stable typed I/O. Register-time assembly rejects incomplete agent contracts. Authors have READ/UPDATE/DECIDE primitives without raw dict control flow.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §12 · §29 · §32.0 · §37.1–2 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 0 |
| ADR | ADR-AGENT-002 · ADR-AGENT-003 (no new ADR) |

## Changed artifacts

- `intergrax/contracts/agent_run.py` — run request/result contracts
- `intergrax/contracts/agent_run_enums.py` — controlled error/terminal/step enums
- `intergrax/contracts/acp_state.py` — `AcpSessionState` envelope
- `intergrax/contracts/agent_step_context.py` — minimal step context
- `intergrax/contracts/agent_contract_section12.py` — §12 defaults
- `intergrax/agents/authoring/step_outcome.py` — StepOutcome factories
- `intergrax/agents/authoring/state_access.py` — typed state helpers
- `intergrax/agents/authoring/state_merge.py` — merge-patch engine
- `intergrax/runtime/registry/agent_assembly_resolver.py` — ACP-CON-4 gate
- `scripts/maintenance/check_agent_typed_state.py` — CI guard
- `tests/unit/contracts/` · `tests/unit/agents/authoring/` — Wave 0 coverage

## Verification

```bash
uv run pytest tests/unit/contracts/ tests/unit/agents/authoring/ tests/unit/runtime/registry/test_agent_assembly_resolver.py -m gate -q
uv run python scripts/maintenance/check_agent_typed_state.py
```

28 Wave 0 tests green; typed-state script OK.

## Risks and follow-ups

- Wave 1 step loop depends on `HarnessKernel.execute_step` policy/trace wiring (cross-domain UAEP/OBS).
- `AgentRunTrace.steps` remains a stub until ACP-OBS-1 (Wave 3).
- Next: `ACP-STEP-1` · `ACP-STEP-2` · `ACP-STEP-2b`
