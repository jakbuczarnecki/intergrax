---
id: IJ-2026-06-11-009
date: 2026-06-11
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-PROD-6
status: completed
commit: pending
adr: none — bridge existing ReliabilityProfile.idempotency_store port to ACP ledger replay
---

# ACP-CLOSE PROD-6 — idempotency store cross-run dedupe

## Operator request

Execute next ACP-CLOSE sprint: wire `ReliabilityProfile.idempotency_store` to declarative side-effect ledger replay per architecture §40.2.2.

## Summary

Added `idempotency_ledger_bridge` and `idempotency_store_wiring` so committed declarative tool keys persist in the host idempotency store and replay-skip across new `run_id` values. Kernel validation and `execute_declarative_actions` consult both in-run ledger and durable store. Wired through `StepKernelContext`, `acp_run`, Nexus `graph_executor`, harness host runtime, task enricher, and Tier-3 factories.

## Project impact

Mutating declarative tools achieve effective exactly-once semantics across separate agent runs when reliability idempotency is enabled. GAP-ACP-26 fully closed.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.2.2 · GAP-ACP-26 |
| Plan | `ACP-CLOSE-PROD-6` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/persistence/idempotency_ledger_bridge.py` — store ↔ ledger bridge (new)
- `intergrax/agents/persistence/idempotency_store_wiring.py` — host metadata wiring (new)
- `intergrax/contracts/acp_metadata_keys.py` — `IDEMPOTENCY_STORE` key
- `intergrax/agents/persistence/tool_action_validation.py` — cross-run replay skip
- `intergrax/agents/persistence/declarative_tool_executor.py` — commit + external_ref hydration
- `intergrax/runtime/kernel/step_kernel.py` — `idempotency_store` on kernel context
- `intergrax/agents/authoring/acp_run.py` — resolve store from metadata
- `intergrax/runtime/nexus/nexus_loop.py`, `graph_executor.py` — Nexus inject
- `intergrax/applications/_shared/harness_host_runtime.py`, `nexus_factory.py`, `task_control_wiring.py`
- `applications/*/host/factory.py` — pass `runtime.reliability.idempotency_store`
- `tests/unit/agents/persistence/test_idempotency_ledger_bridge.py` (new)
- `tests/acceptance/agent_os/test_acp_declarative_mutating_cross_run_dedupe.py` (new)

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_idempotency_ledger_bridge.py tests/acceptance/agent_os/test_acp_declarative_mutating_cross_run_dedupe.py tests/unit/applications/test_acp_checkpoint_host_wiring.py -q
```

Result: 18 passed.

## Risks and follow-ups

- Host idempotency store is in-memory unless `idempotency_db_path` is set on `build_harness_host_runtime`.
- Compensation queue background drain (PROD-5 follow-up) remains separate.
