---
id: IJ-2026-06-18-002
date: 2026-06-18
tiers:
  - tier-1
  - tier-2
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - EBE-8
status: completed
commit: ae4b5680
adr: docs/project/technical/adr/entries/2026-06-13/ADR-OBS-002.md — EBE-8 folded into accepted ADR scope; no new ADR
---

# EBE-8 — PoC v2 harness_step boundary export for AgentReceipt partner

## Operator request

Implement the full PoC v2 iteration agreed with partner Cullen: keep unsigned `client_observed` trust model and `boundary_events[]` API delivery; add `harness_step` events alongside `tool_execution`; assign stable `event_id` plus monotonic `event_sequence` per run; enable one receipt per boundary event; update docs, tests, and commit.

## Summary

Extended Execution Boundary Export with HarnessKernel step-level emission (`HarnessBoundaryEmitter`), schema v2 fields (`boundary_type`, `event_sequence`, `policy_verdicts`, `step_outcome`), buffer sequencing, `step_level_enabled` on `ExecutionBoundaryExportProfile`, UAEP/ACP kernel wiring, and `attestation_demo` v2 contract tests expecting two ordered events per run.

## Project impact

Partners can sign separate receipts for tool execution and harness step completion within the same run, grouped by `run_id` and ordered by `event_sequence`, without changing the HOS spine or trust role.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §18 |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` EBE-8 |
| Application | `applications/attestation_demo/docs/ARCHITECTURE.md` |
| ADR | `docs/project/technical/adr/entries/2026-06-13/ADR-OBS-002.md` |

## Changed artifacts

- `intergrax/runtime/attestation/execution_boundary_event.py` — v2 schema fields
- `intergrax/runtime/attestation/harness_boundary_emitter.py` — kernel-side emitter
- `intergrax/runtime/attestation/buffer.py` — `event_sequence` assignment
- `intergrax/runtime/kernel/step_kernel.py` — `_finish_step` hook
- `intergrax/agents/uaep.py`, `acp_run.py` — kernel EBE wiring
- `applications/attestation_demo/` — profile, tests, partner handoff

## Verification

```bash
uv run pytest tests/unit/runtime/attestation/ applications/attestation_demo/attestation_demo_tests -q
```

Result: 26 passed.

## Risks and follow-ups

- EBE-7 webhook sink and EBE-9 host signing remain deferred.
- Partner adapter must loop `boundary_events[]` (one receipt per element).
- Merge to `agent_experiment_runtime` for Cullen handoff when operator requests.
