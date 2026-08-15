---
id: IJ-2026-06-17-008
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.2
  - OBS-EVOL-9.3
status: completed
commit: 6568ba48
adr: ADR-OBS-003
---

# OBS-EVOL-9.2/9.3 — RuntimeEvent taxonomy fields and public emit APIs

## Operator request

Continue OBS-EVOL-9 Layer Completion after SAR acceptance — implement taxonomy fields and public emit APIs.

## Summary

Added `event_kind`, `event_category`, `ops_hint` on `RuntimeEvent` with catalog auto-fill via `model_post_init`; introduced `DOMAIN_SIGNAL` spine carrier, `EmitContext`, `emit_domain_signal` (production redaction), and `emit_platform_event`. Extracted `event_taxonomy.py` to break import cycles.

## Project impact

Tier-2/3 authors can emit domain signals without new spine enum members; platform events get consistent taxonomy metadata on the bus.

## Traceability

| Link | Target |
|------|--------|
| ADR | `docs/project/technical/adr/entries/2026-06-17/ADR-OBS-003.md` |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` OBS-EVOL-9.2 · 9.3 |
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §4.4.4 · §4.4.8 · §4.4.9 |

## Changed artifacts

- `intergrax/runtime/events/runtime_event.py`
- `intergrax/runtime/events/event_taxonomy.py`
- `intergrax/runtime/events/emit_context.py`
- `intergrax/runtime/events/signals.py`
- `intergrax/runtime/events/event_catalog.py`
- `intergrax/runtime/events/payloads/base.py`
- `intergrax/runtime/events/__init__.py`
- `tests/unit/runtime/events/test_runtime_event_kind.py`
- `tests/unit/runtime/events/test_domain_signals.py`

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_runtime_event_kind.py tests/unit/runtime/events/test_domain_signals.py tests/unit/runtime/events/test_event_catalog.py -q
```

## Risks and follow-ups

- OBS-EVOL-9.4 EventKindRegistry not yet enforcing registered kinds at emit time.
- Bus taxonomy subscribe (9.5) and CI `check_event_catalog.py` (9.6) still open.
- Spine consolidation (9.7) remains before external publication.
