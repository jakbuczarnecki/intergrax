---
id: IJ-2026-06-10-019
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CON-6
  - ACP-CON-7
status: completed
commit: pending
adr: none — enforcement of existing §37.6 routing invariant
---

# Enforce capability routing and agent step security gates

## Operator request

Execute the next ACP sprint: Wave 6 routing and security foundations (ACP-CON-6, ACP-CON-7).

## Summary

Added typed task routing contract rejecting class-name and import-path routing keys. Centralized capability resolution in `capability_routing.py` and wired `AgentRouter` to validate tasks and select best capability match. Added CI scripts `check_capability_routing.py` and `check_agent_step_security.py` with unit tests for dual-implementation capability selection.

## Project impact

Nexus enforces capability-token routing at runtime; task payloads cannot smuggle Python class names. Tier-2 agent entry modules have a dedicated security static gate complementing vendor-import checks.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §37.6–§37.7 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 6 steps 6.1–6.2 |
| Audit map | Layer 9 — Agent contracts |

## Changed artifacts

- `intergrax/contracts/task_routing.py` — forbidden routing keys contract
- `intergrax/runtime/registry/capability_routing.py` — capability match selection
- `intergrax/runtime/nexus/agent_router.py` — validation + best-match wiring
- `scripts/maintenance/check_capability_routing.py`, `scripts/maintenance/check_agent_step_security.py` — CI gates
- `tests/unit/runtime/registry/test_capability_routing_acp_con6.py` — routing tests

## Verification

```bash
uv run python scripts/maintenance/check_capability_routing.py
uv run python scripts/maintenance/check_agent_step_security.py
uv run pytest tests/unit/runtime/registry/test_capability_routing_acp_con6.py -q
```

Result: pass (4 tests).

## Risks and follow-ups

- ACP-ORG-1..5 organizational policy envelope remains open (Wave 6 steps 6.3–6.5).
- `run_pipeline_step` still allowed in scaffold `steps/pipeline.py` stubs until ACP-LEG full removal.
- STRICT profile widen deny documented as follow-up in readiness security dimension.
