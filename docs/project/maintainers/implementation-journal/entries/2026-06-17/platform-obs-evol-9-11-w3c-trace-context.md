---
id: IJ-2026-06-17-014
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.11
status: completed
commit: d2e5546e
adr: ADR-OBS-003
---

# OBS-EVOL-9.11 — W3C Trace Context on RuntimeEvent

## Operator request

Continue OBS-EVOL-9 — W3C `traceparent` / `tracestate` (SAR-04).

## Summary

Added optional `traceparent` and `tracestate` on `RuntimeEvent`, W3C helpers, `EmitContext` propagation, Nexus publisher injection from task metadata, and OTLP journal export preferring W3C trace/span ids.

## Project impact

Tier-3 hosts can correlate Harness journal events with external APM via standard W3C headers on inbound tasks and exported OTLP snapshots.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §4.4.12 |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` OBS-EVOL-9.11 |

## Changed artifacts

- `intergrax/runtime/events/w3c_trace_context.py`
- `intergrax/runtime/events/runtime_event.py`
- `intergrax/runtime/events/emit_context.py`
- `intergrax/runtime/events/signals.py`
- `intergrax/runtime/nexus/orchestration/task_events.py`
- `intergrax/runtime/observability/journal_export.py`
- `intergrax/runtime/observability/export_bridge.py`
- `tests/unit/runtime/events/test_w3c_trace_context.py`
- `docs/project/architecture/OBSERVABILITY.md`
- `docs/project/maintainers/plans/OBSERVABILITY.md`

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_w3c_trace_context.py -q
uv run python scripts/maintenance/check_event_catalog.py
```

## Risks and follow-ups

- HTTP middleware auto-capture of inbound `traceparent` headers is host-specific (store on `task.metadata`).
- OBS-EVOL-9.9 optional `runtime_event.v2` remains low-priority backlog.
