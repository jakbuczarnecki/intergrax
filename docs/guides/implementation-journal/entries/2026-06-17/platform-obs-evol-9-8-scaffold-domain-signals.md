---
id: IJ-2026-06-17-012
date: 2026-06-17
tiers:
  - tier-0
  - tier-2
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.8
status: completed
commit: 7a81df64
adr: ADR-OBS-003
---

# OBS-EVOL-9.8 — Scaffold domain signal templates

## Operator request

Continue OBS-EVOL-9 — Sprint S7 (scaffold `emit_domain_signal`).

## Summary

Added `signal_templates.py` with agent/application `signals/` scaffold (`example_signal`, `registry`, `emit`), wired into `new_agent` and `new_application`, and helper IDs in `extension_sdk` (`agent_signal_*`, `application_signal_*`).

## Project impact

New agents and applications get a working `emit_domain_signal` pattern with registered `event_kind` and typed payload out of the box.

## Traceability

| Link | Target |
|------|--------|
| Plan | `docs/plan/OBSERVABILITY.md` OBS-EVOL-9.8 |
| Guides | `AGENT_CREATION_GUIDE.md` §Q.5 · `EXTENSION_AUTHOR_GUIDE.md` §11 |

## Changed artifacts

- `intergrax/scaffold/signal_templates.py`
- `intergrax/scaffold/new_agent.py`
- `intergrax/scaffold/new_application.py`
- `intergrax/runtime/observability/extension_sdk.py`
- `tests/unit/scaffold/test_scaffold_domain_signals.py`
- `docs/plan/OBSERVABILITY.md`

## Verification

```bash
uv run pytest tests/unit/scaffold/test_scaffold_domain_signals.py -q
```

## Risks and follow-ups

- Authors must call `register_signal_schemas()` at host bootstrap (same pattern as tracing registry).
- OBS-EVOL-9.10 declarative `ObservabilityProfile` subscriptions remain optional P2.
