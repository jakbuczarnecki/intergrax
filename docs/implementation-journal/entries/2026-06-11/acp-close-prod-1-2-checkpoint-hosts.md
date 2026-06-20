---
id: IJ-2026-06-11-004
date: 2026-06-11
tiers:
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-PROD-1
  - ACP-CLOSE-PROD-2
status: completed
commit: pending
adr: none — composition of existing ACP-PROD-1 modules on harness hosts; no new semantics
---

# ACP-CLOSE PROD-1/2 — agent checkpoint store on product harness hosts

## Operator request

Complete sprint 3 of ACP-CLOSE: wire `AgentCheckpointStore` and `acp_checkpoint_task_enricher` on all Tier-3 harness product hosts (lab pattern).

## Summary

Added `acp_checkpoint_host_wiring.py` to resolve durable `agent_checkpoints.db` adjacent to task checkpoints. `build_harness_host_runtime` now auto-materializes `agent_checkpoint_store` and exposes it on `HarnessHostRuntime`. `build_reliability_task_enricher` chains the ACP checkpoint enricher when a store is provided. Updated legal, research, assistant, dispute_sim, local_workspace, and poc_template factories.

## Project impact

All harness product hosts now inject agent checkpoint metadata into Nexus tasks — prerequisite for mutating-agent resume on production paths. GAP-ACP-25 fully closed.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.1 · GAP-ACP-25 |
| Plan | `ACP-CLOSE-PROD-1` · `ACP-CLOSE-PROD-2` |
| ADR | none |

## Changed artifacts

- `intergrax/applications/_shared/acp_checkpoint_host_wiring.py` — store path resolution (new)
- `intergrax/applications/_shared/harness_host_runtime.py` — auto-resolve + expose store
- `intergrax/applications/_shared/task_control_wiring.py` — ACP enricher in reliability chain
- `applications/*/host/factory.py` — six harness product hosts
- `tests/unit/applications/test_acp_checkpoint_host_wiring.py` — wiring unit tests (new)
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — PROD-1/2 Done
- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — GAP-ACP-25 Closed

## Verification

```bash
uv run pytest tests/unit/applications/test_acp_checkpoint_host_wiring.py tests/unit/applications/test_harness_reliability_wiring.py -q
```

Result: 13 passed.

## Risks and follow-ups

- `CatalogDeclarativeToolInvoker` still uses mock shim (ACP-CLOSE-PROD-3/4).
- Scoreboard mutating dimensions still below 100% (ACP-CLOSE-PROD-8).
- Durable compensation queue and idempotency store depth remain open (PROD-5/6).
