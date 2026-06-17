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
commit: pending
adr: ADR-OBS-003
---

# OBS-EVOL-9.2/9.3 — RuntimeEvent taxonomy fields and public emit APIs

## Summary

Added `event_kind`, `event_category`, `ops_hint` on `RuntimeEvent` with catalog auto-fill; introduced `DOMAIN_SIGNAL` spine carrier, `EmitContext`, `emit_domain_signal` (production redaction), and `emit_platform_event`.

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_runtime_event_kind.py tests/unit/runtime/events/test_domain_signals.py -q
```
