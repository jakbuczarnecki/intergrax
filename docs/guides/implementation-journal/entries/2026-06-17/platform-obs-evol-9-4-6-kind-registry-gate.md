---
id: IJ-2026-06-17-009
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.4
  - OBS-EVOL-9.6
status: completed
commit: 5fbeda02
adr: ADR-OBS-003
---

# OBS-EVOL-9.4/9.6 — EventKindRegistry and catalog CI gate

## Operator request

Continue OBS-EVOL-9 Layer Completion — Sprint S4 (kind registry + catalog gate).

## Summary

Shipped `EventKindRegistry` with payload schema binding, wired `emit_domain_signal` to require registered kinds, deterministic persistence sampling via `should_persist_event`, `check_event_catalog.py` in observability gates, and SAR-08 LLM stream namespace separation test.

## Project impact

Domain signals are enforceable at emit time; high-volume spine events sample at persistence boundary without losing in-memory history.

## Traceability

| Link | Target |
|------|--------|
| Plan | `docs/plan/OBSERVABILITY.md` OBS-EVOL-9.4 · 9.6 |
| Architecture | `docs/architecture/OBSERVABILITY.md` §4.4.7 · EVT-AP-06 |

## Changed artifacts

- `intergrax/runtime/events/event_kind_registry.py`
- `intergrax/runtime/events/event_catalog.py` (`should_persist_event`)
- `intergrax/runtime/events/event_bus.py`
- `intergrax/runtime/events/signals.py`
- `intergrax/runtime/observability/extension_sdk.py`
- `scripts/check_event_catalog.py`
- `scripts/check_observability_gates.py`
- `tests/unit/runtime/events/test_event_kind_registry.py`
- `tests/unit/runtime/events/test_event_bus_sampling.py`

## Verification

```bash
uv run python scripts/check_event_catalog.py
uv run pytest tests/unit/runtime/events/test_event_kind_registry.py tests/unit/runtime/events/test_event_bus_sampling.py -q
```

## Risks and follow-ups

- OBS-EVOL-9.5 bus taxonomy subscribe and JournalQuery still open.
- Spine consolidation 9.7 remains before external publication.
