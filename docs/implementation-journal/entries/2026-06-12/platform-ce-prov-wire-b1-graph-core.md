---
id: IJ-2026-06-12-024
date: 2026-06-12
tiers:
  - tier-0
  - tier-1
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-PROV-BRIDGE
  - CE-PROV-01
  - CE-PROV-03
  - CE-PROV-08
status: completed
commit: pending
adr: none — adapter wiring only; no new cross-cutting contract
---

# CE-PROV-WIRE B1 — legacy bridge and graph core builtin providers

## Operator request

Continue layer completion iteratively after TOOLS closeout: pick the first incomplete harness layer and implement the next CE-PROV-WIRE sprint (graph core provider wiring).

## Summary

Shipped `legacy_bridge.py` with handle-key contract and fragment adapters for task message, graph prior outputs, and session history. Wired `builtin.task_message`, `builtin.graph_prior`, and `builtin.session_history` collectors in `BuiltinContextPlugin`. Extended `build_graph_provider_handles` and `ContextManager.build_agent_context_async` to pass `prior_output_records` and raw task messages on the engine path.

## Project impact

Graph `DefaultNexusContextEngine.assemble()` now emits provenance-tagged fragments for task objective, dependency priors, and session turns when handles are populated — first concrete step toward closing GAP-CTX-20 without duplicating composed legacy message text in the user turn.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CONTEXT_ENGINEERING.md` §8.4, §17 |
| Plan | `docs/plan/CONTEXT_ENGINEERING.md` CE-PROV-WIRE sprint B1 |
| ADR | none — reuse existing CE contracts |
| Audit / gap | GAP-CTX-20 partial (8 stubs remain) |

## Changed artifacts

- `intergrax/context/providers/legacy_bridge.py` — shared fragment adapters + handle contract
- `intergrax/context/providers/builtin.py` — live collectors for B1 providers
- `intergrax/runtime/nexus/context/provider_handles.py` — prior/session handle keys
- `intergrax/runtime/nexus/context/context_manager.py` — engine path handle wiring
- `tests/unit/context/test_legacy_bridge_providers.py` — B1 gate tests
- `docs/plan/CONTEXT_ENGINEERING.md`, `docs/architecture/CONTEXT_ENGINEERING.md` — B1 status

## Verification

- `uv run pytest tests/unit/context/test_legacy_bridge_providers.py -q` — 5 passed
- `uv run pytest tests/unit/context/ tests/unit/runtime/nexus/context/test_context_manager_engine.py -q` — 36 passed
- `python scripts/check_context_tier0_import_boundary.py` — OK

## Risks and follow-ups

- UAEP session path still lacks automatic `session_history_messages` from `HistoryLayer` — B2+ or UAEP handle extension.
- Remaining eight builtin stubs (RAG, LTM, tools, policy, etc.) tracked in CE-PROV-WIRE sprints B2–B4.
- `CE-10.3` metadata follow-up deferred until CE-PROV-05/08 fragment shapes validated in integration.
