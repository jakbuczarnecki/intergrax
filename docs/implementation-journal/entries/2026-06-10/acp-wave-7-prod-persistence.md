---
id: IJ-2026-06-10-021
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-1
  - ACP-PROD-2
  - ACP-PROD-3
status: completed
commit: pending
adr: none — implements architecture §40.1–§40.3 persistence contracts
---

# Agent checkpoint store, idempotency ledger, and tool execution profiles

## Operator request

Execute the next ACP sprint: Wave 7 production persistence foundations (ACP-PROD-1 through ACP-PROD-3).

## Summary

Added agent run checkpoint store with in-memory and SQLite backends, side-effect ledger with idempotency dedupe, `ToolExecutionProfile` derived from tool contracts, kernel validation for mutating declarative actions, and ACP session resume wiring via checkpoint metadata keys.

## Project impact

Typed agent runs can checkpoint after successful steps, resume from durable state, and block mutating tools without idempotency keys — unblocking mutating-agent staging gate per §40.12.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.1–§40.3 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 7 steps 7.1–7.3 |
| Cross-domain | `RELIABILITY_FAILURE_AND_HITL`, `TOOLS` |

## Changed artifacts

- `intergrax/contracts/side_effect.py`
- `intergrax/agents/persistence/checkpoint_store.py`, `side_effect_ledger.py`, `session_persistence.py`, `tool_action_validation.py`
- `intergrax/tools/tool_execution_profile.py`
- `intergrax/runtime/kernel/step_kernel.py`, `intergrax/agents/authoring/acp_run.py`
- `tests/unit/agents/persistence/test_acp_prod_persistence.py`

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_acp_prod_persistence.py tests/unit/runtime/kernel/test_step_kernel.py -q -m gate
```

Result: pass.

## Risks and follow-ups

- ACP-PROD-4..11 remain open (reliability profile in kernel, shared context CAS, CI matrix).
- Compensation tool enqueue not implemented — profile metadata only.
- Hosts must inject `AgentCheckpointStore` via `AcpMetadataKey.CHECKPOINT_STORE` for durable resume.
