---
id: IJ-2026-06-17-011
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.7
status: completed
commit: 39ee2f6f
adr: ADR-OBS-003
---

# OBS-EVOL-9.7 — Pre-release spine consolidation

## Operator request

Continue OBS-EVOL-9 — Sprint S6 (spine consolidation).

## Summary

Removed 19 legacy `RuntimeEventType` members (adaptive, capacity/scale, autonomy, recovery, hook). Platform emitters now use `build_platform_signal_event()` on `DOMAIN_SIGNAL` with `platform.*` kinds. Added `spine_consolidation.py`, read-path `migrate_legacy_spine_payload()`, publication budget gate (56 types), and updated unit tests.

## Project impact

Spine enum is publication-ready at 56 members. Adaptive, capacity, and hook observability signals use unified domain-signal identity without enum churn.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/OBSERVABILITY.md` — §4.4.13 |
| Plan | `docs/plan/OBSERVABILITY.md` — OBS-EVOL-9.7 |
| ADR | `docs/adr/entries/2026-06-17/ADR-OBS-003.md` |

## Changed artifacts

- `intergrax/runtime/events/spine_consolidation.py`
- `intergrax/runtime/events/runtime_event.py`
- `intergrax/runtime/events/event_catalog.py`
- `intergrax/runtime/adaptive/adaptive_runtime_events.py`
- `intergrax/runtime/capacity/events.py`
- `intergrax/runtime/middleware/hook_runtime_guard.py`
- `scripts/maintenance/check_event_catalog.py`
- `tests/unit/runtime/events/test_spine_consolidation.py`
- `tests/unit/runtime/adaptive/test_adaptive_apply_wave4.py`
- `tests/unit/runtime/capacity/test_capacity_events_gate.py`
- `tests/unit/runtime/middleware/test_hook_runtime_guard.py`
- `tests/unit/runtime/events/test_event_catalog.py`
- `docs/architecture/OBSERVABILITY.md`
- `docs/plan/OBSERVABILITY.md`

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_spine_consolidation.py \
  tests/unit/runtime/adaptive/test_adaptive_apply_wave4.py \
  tests/unit/runtime/capacity/test_capacity_events_gate.py \
  tests/unit/runtime/middleware/test_hook_runtime_guard.py -q
uv run python scripts/maintenance/check_event_catalog.py
```

## Risks and follow-ups

- OBS-EVOL-9.8 scaffold templates (`emit_domain_signal` in `new_agent` / `new_application`) is next in PR order.
- Pre-existing `test_observability_l4_platform_bootstrap_registers_journal_export` fails without `fastmcp` server extra (unrelated).
