---
id: IJ-2026-06-17-043
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: LLM_ADAPTERS
plan_ref:
  - M-LLM-X.4.4
status: completed
commit: pending
adr: none — observability DTO on existing failover path; no contract or tier boundary change
---

# M-LLM-X.4.4 — LLM failover routing attempt trace diagnostics

## Operator request

Execute Harness Architecture Audit Mode B (implement plan backlog) and deliver the next open P1 slice on the LLM adapters domain — failover observability per profile attempt.

## Summary

Implemented `LLMRoutingAttemptDiagV1` (schema `intergrax.diag.engine.core_llm.routing_attempt`) and Tier-1 bridge `attach_failover_routing_trace_observer`. `FailoverLLMAdapter` now carries stable `profile_id` labels from `ModelRouter.ordered_profile_ids()`, clears per-call attempt records, and invokes an optional observer on each retriable failure. `RuntimeState.configure_llm_tracker()` wires the observer to the core adapter trace spine. `trace_bridge` maps the new schema to `LLM_CALL` runtime events.

## Project impact

Operators and replay tooling can see which LLM profile failed before failover without scraping adapter logs. Closes the observability gap on M-LLM-X wave 4; Tier-3 fallback list wiring (M-LLM-X.4.5) remains backlog.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/LLM_ADAPTERS.md` §Routing |
| Plan | `docs/plan/LLM_ADAPTERS.md` M-LLM-X.4.4 |
| ADR | none — extends existing failover adapter (ADR-LLM-002) |

## Changed artifacts

- `intergrax/llm_adapters/registry/failover_adapter.py` — profile_id, observer, per-call attempt reset
- `intergrax/llm_adapters/registry/model_router.py` — `ordered_profile_ids()`
- `intergrax/llm_adapters/registry/profile.py` — pass profile ids into failover wrapper
- `intergrax/runtime/nexus/tracing/adapters/llm_routing_attempt.py` — diag DTO + attach helper
- `intergrax/runtime/nexus/engine/runtime_state.py` — wire observer on tracker setup
- `intergrax/runtime/events/trace_bridge.py` — schema → `LLM_CALL` mapping
- `docs/plan/LLM_ADAPTERS.md` — M-LLM-X.4.4 Done

## Verification

```bash
uv run pytest tests/unit/llm_adapters/test_failover_adapter.py tests/unit/llm_adapters/test_llm_routing_attempt_trace.py tests/unit/runtime/events/test_trace_bridge.py -q
uv run python scripts/check_trace_bridge_event_catalog.py
```

Result: 9 passed; trace bridge catalog OK.

## Risks and follow-ups

- M-LLM-X.4.5 Tier-3 `ApplicationEnvironmentProfile` fallback list wiring still open (P1 partial wave X-4).
- Tool-planner / websearch failover adapters are not auto-wired; only core `llm_adapter` on `RuntimeState`.
