---
id: IJ-2026-06-17-007
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.1
  - OBS-EVOL-9-SAR
status: completed
commit: 14094dda
adr: ADR-OBS-003
---

# OBS-EVOL-9.1 — EventCatalog SSOT + SAR architecture fold-in

## Operator request

Layer Completion Mode on OBSERVABILITY: accept all SAR proposals, update architecture/plan, implement OBS-EVOL-9.1.

## Summary

Folded accepted SAR items (EmitContext, retention_class, sampling, JournalQuery, profile subscriptions, traceparent, deprecation shim, LLM namespace, domain redaction) into architecture §4.4.7–4.4.13 and plan register OBS-EVOL-9.10–9.11. Shipped `event_catalog.py` as single source of truth merging phase, ops hint, category, payload schema, sample_rate, retention_class, and consolidation_kind; `phase_coverage.py` is now a deprecated view.

## Project impact

Gates and emitters can migrate to one catalog; spine consolidation (9.7) has `consolidation_kind` metadata pre-declared.

## Traceability

| Link | Target |
|------|--------|
| ADR | `docs/project/technical/adr/entries/2026-06-17/ADR-OBS-003.md` |
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §4.4.7–4.4.13 |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` OBS-EVOL-9 |

## Changed artifacts

- `docs/project/architecture/OBSERVABILITY.md`
- `docs/project/maintainers/plans/OBSERVABILITY.md`
- `docs/project/technical/adr/entries/2026-06-17/ADR-OBS-003.md`
- `intergrax/runtime/events/event_catalog.py`
- `intergrax/runtime/events/phase_coverage.py`
- `intergrax/runtime/events/__init__.py`
- `tests/unit/runtime/events/test_event_catalog.py`

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_event_catalog.py tests/unit/runtime/events/test_trace_bridge_event_catalog.py -q
uv run python scripts/maintenance/check_observability_gates.py
python scripts/maintenance/check_implementation_journal.py
```

## Risks and follow-ups

- OBS-EVOL-9.2–9.7 still open — `RuntimeEvent` lacks `event_kind` until 9.2.
- `check_event_catalog.py` CI gate ships in 9.6.
- L4 depth gate `test_observability_l4_platform_bootstrap_registers_journal_export` fails on missing FastMCP server extra (pre-existing env).
