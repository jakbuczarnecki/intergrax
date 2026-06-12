---
id: IJ-2026-06-12-010
date: 2026-06-12
tiers:
  - tier-0
  - tier-1
  - tier-2
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-4.1
  - CE-4.2
  - CE-4.3
  - CE-4.4
  - CE-4.5
  - CE-4.6
  - CE-4.7
  - CE-5.1
status: completed
commit: 269d64c4
adr: no ADR needed — extends CE contracts and event payloads without Nexus semantic change
---

# CE Sprint 5 — Step-aware context assembly

## Operator request

Continue CE sprint batch S5–S12; deliver step-aware assembly (S5).

## Summary

Added `DefaultContextRanker`, `run_pre_context_policy_gate`, `AgentContextHints` on contracts, ACP assembly bridge, `ContextAssemblyPayloadV2`, and engine collect/rank/policy integration.

## Project impact

ACP steps carry `step_kind` / `step_index`; graph and assembly events can emit v2 payloads with step metadata.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CONTEXT_ENGINEERING.md` |
| Plan | CE-4.1–CE-4.7, CE-5.1, Sprint S5 |

## Changed artifacts

- `intergrax/context/ranker.py` — step-kind source boosts
- `intergrax/runtime/policy/context_assembly_policy.py` — pre/post collect gate
- `intergrax/agents/authoring/context_assembly_bridge.py` — ACP request builder
- `intergrax/runtime/events/payloads/canonical.py` — `context_assembly.v2`

## Verification

```bash
uv run pytest tests/unit/context/test_step_aware_ranker.py tests/unit/runtime/nexus/context/ -m gate -q
```

Result: 29 passed.

## Risks and follow-ups

- Episodic vector recall deferred to S6 (CE-VEC-1).
