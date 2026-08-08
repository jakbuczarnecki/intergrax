---
id: IJ-2026-06-11-003
date: 2026-06-11
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-LEG-1
  - ACP-CLOSE-LEG-2
status: completed
commit: pending
adr: none — removal of deprecated bridge per plan ACP-CLOSE; no new semantics
---

# ACP-CLOSE LEG-1/2 — remove RuntimeEngine fallback and linear UAEP author API

## Operator request

Execute sprint 2 of ACP-CLOSE: remove legacy `RuntimeEngine` fallback from `AgentEngine` and move linear UAEP `decide_after_step` off the public `IntergraxAgent` surface.

## Summary

`AgentEngine` now raises `ValueError` (ACP-CLOSE-LEG-1) when an agent is neither ACP-session nor `UAEPAgent`. Added `uaep_linear_bridge.py` with `linear_agent_get_steps` / `linear_agent_decide_after_step`; removed `decide_after_step` from `IntergraxAgent`; `UAEPExecutor._decide_after_step` routes linear agents through the bridge. Integration tests updated to reject pipeline-only agents or use explicit UAEP stubs.

## Project impact

DEBT-ACP-06 and DEBT-ACP-04 closed for the linear author path. Single typed execution entry enforced at `AgentEngine` — no silent `RuntimeEngine.run` fallback.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §13.5 · GAP-ACP-03 partial |
| Plan | `ACP-CLOSE-LEG-1` · `ACP-CLOSE-LEG-2` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/agent_engine.py` — hard reject non-UAEP/non-ACP
- `intergrax/agents/authoring/uaep_linear_bridge.py` — internal UAEP linear bridge (new)
- `intergrax/agents/authoring/base.py` — no `decide_after_step`; delegate get_steps to bridge
- `intergrax/agents/uaep.py` — linear decide routing
- `tests/unit/agents/test_agent_engine_legacy_deprecation.py` — expect ValueError
- `tests/unit/agents/authoring/test_uaep_linear_bridge.py` — bridge unit test (new)
- `tests/integration/agents/test_agent_engine_*.py` — align with LEG-1

## Verification

```bash
uv run pytest tests/unit/agents/test_agent_engine_legacy_deprecation.py tests/unit/agents/authoring/test_uaep_linear_bridge.py tests/integration/agents/ tests/unit/agents/test_uaep_executor.py -q
```

Result: pass.

## Risks and follow-ups

- `uaep_pipeline.run_pipeline_step` still uses `RuntimeEngine` (ACP-CLOSE-LEG-3).
- Product hosts still need checkpoint wiring (ACP-CLOSE-PROD-1/2).
