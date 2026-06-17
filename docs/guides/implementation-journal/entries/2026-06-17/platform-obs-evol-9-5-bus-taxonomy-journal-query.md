---
id: IJ-2026-06-17-010
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.5
status: completed
commit: de515ae7
adr: ADR-OBS-003
---

# OBS-EVOL-9.5 — Bus taxonomy subscribe and JournalQuery

## Operator request

Continue OBS-EVOL-9 — Sprint S5.

## Summary

Extended `RuntimeEventBus.subscribe` with `categories`, `kind_prefix`, and `ops_hints`; taxonomy handlers dispatch on both `publish` and `record`. Added `query_journal()` read-model filter.

## Project impact

Tier-3 hooks can subscribe by `kind_prefix` without enum lists; debug tooling can filter journals by taxonomy.

## Traceability

| Link | Target |
|------|--------|
| Plan | `docs/plan/OBSERVABILITY.md` OBS-EVOL-9.5 |
| Architecture | `docs/architecture/OBSERVABILITY.md` §4.4.5 · §4.4.10 |

## Changed artifacts

- `intergrax/runtime/events/event_bus.py`
- `intergrax/runtime/events/journal_query.py`
- `scripts/check_event_catalog.py`
- `tests/unit/runtime/events/test_event_bus_taxonomy_subscribe.py`

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_event_bus_taxonomy_subscribe.py -q
```

## Risks and follow-ups

- Spine consolidation OBS-EVOL-9.7 remains the main P1 item before publication.
