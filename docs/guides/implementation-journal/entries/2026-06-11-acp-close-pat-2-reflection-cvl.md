---
id: IJ-2026-06-11-016
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
  - tier-3
scope: AGENT_CONTRACTS, CRITIC_VERIFICATION
plan_ref:
  - ACP-CLOSE-PAT-2
status: completed
commit: pending
adr: none — composes existing CVL `validate_uaep_step_with_critic_detail`; gateway only
---

# ACP-CLOSE PAT-2 — ReflectionAgent CVL critic gateway

## Operator request

Continue ACP-CLOSE sprint: wire `ReflectionAgent` to CVL critic hooks via gateway only (no critic SDK in Tier-2).

## Summary

Added `critic_gateway.verify_reflection_draft` (ACP session metadata → `CriticGraphHooks` → CVL partial verify). `ReflectionAgent` invokes CVL at `critique` phase when draft exists and hooks are wired. Extended `ACPSessionHostContext` + `acp_run` to pass `critic_graph_hooks`; `build_acp_session_host_from_harness` helper for Tier-3 hosts. `ReflectionSessionState` gains reflection round budgets.

## Project impact

Reflection pattern can use host-configured CVL without Tier-2 critic imports; legal/compliance hosts enable critic via existing `CriticProfile` wiring.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `AGENT_CONTRACTS_AND_ASSEMBLY` §26.6 |
| Plan | `ACP-CLOSE-PAT-2` |
| ADR | none — reuses CRIT-V CVL stack |

## Changed artifacts

- `intergrax/agents/authoring/critic_gateway.py` (new)
- `intergrax/agents/authoring/patterns/reflection.py`
- `intergrax/agents/authoring/acp_run.py`, `acp_session_host.py`
- `intergrax/applications/_shared/acp_session_host_wiring.py` (new)
- `intergrax/contracts/acp_metadata_keys.py` — `CRITIC_HOOKS`, `TENANT_ID`
- `intergrax/agents/authoring/patterns/states.py` — reflection rounds

## Verification

```bash
uv run pytest tests/unit/agents/authoring/test_critic_gateway.py tests/unit/agents/authoring/patterns/test_reflection_critic.py tests/unit/applications/test_acp_session_host_wiring.py -m gate -q
```

## Risks and follow-ups

- ACP-CLOSE-CI-2 (anti-pattern ACP-AP-02 gate) remains open.
- Hosts must pass `ACPSessionHostContext` with critic hooks for direct `agent.run()` — Nexus graph path uses existing UAEP critic hooks.
