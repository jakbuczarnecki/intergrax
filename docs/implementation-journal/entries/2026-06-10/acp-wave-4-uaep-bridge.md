---
id: IJ-2026-06-10-013
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-STEP-3
  - ACP-LEG-1
  - ACP-LEG-3
status: completed
commit: pending
adr: none — compatibility bridge per architecture §13.4–§38; no contract change
---

# ACP Wave 4 — UAEP kernel bridge and RuntimeEngine deprecation

## Operator request

Continue Phase ACP sprint: Wave 4 legacy bridge so existing UAEP agents run through HarnessKernel without rewrite, plus deprecate the RuntimeEngine author fallback.

## Summary

Added `uaep_step_bridge` translating `AgentDecision` → `StepOutcome` and routing each `UAEPExecutor.execute_step` through `HarnessKernel.execute_step` with typed metadata keys. `AgentEngine` emits `DeprecationWarning` on `RuntimeEngine` fallback. `RuntimeEngine` module docstring marked INTERNAL ONLY.

## Project impact

Legacy roster agents gain Plane B kernel trace and policy enforcement without migration; new authors are steered away from `RuntimeEngine` toward `on_next_step` / `Agent.run(AgentRunRequest)`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §13.4, §38 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1aw Wave 4 |

## Changed artifacts

- `intergrax/agents/authoring/uaep_step_bridge.py` — UAEP ↔ kernel bridge
- `intergrax/contracts/uaep_bridge_keys.py` — typed session keys
- `intergrax/agents/uaep.py` — kernel session per run; bridged `execute_step`
- `intergrax/agents/agent_engine.py` — deprecation warning
- `intergrax/runtime/nexus/engine/runtime.py` — internal-only docstring

## Verification

```bash
uv run pytest tests/unit/agents/test_uaep_executor.py tests/unit/agents/authoring/test_uaep_step_bridge.py tests/unit/agents/test_agent_engine_legacy_deprecation.py -q
```

Result: 7 passed.

## Risks and follow-ups

- UAEP resume path still calls `run_step` directly in `_execute_step_with_resume` — extend bridge in Wave 5 if needed.
- Wave 5: cognitive patterns + scaffold without UAEP boilerplate (`ACP-LEG-4`).
