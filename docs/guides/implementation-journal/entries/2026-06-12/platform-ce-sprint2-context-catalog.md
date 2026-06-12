---
id: IJ-2026-06-12-007
date: 2026-06-12
tiers:
  - tier-0
  - tier-3
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-1.5
  - CE-2.1
  - CE-2.2
  - CE-2.3
  - CE-2.4
  - CE-2.5
status: completed
commit: pending
adr: no ADR needed — catalog bootstrap follows ADR-CTX-001 plugin surface
---

# CE Sprint 2 — Context plugin catalog bootstrap

## Operator request

Continue CE sprint workflow: implement plugin catalog bootstrap after Tier-0 contracts (S2).

## Summary

Shipped `bootstrap_context_catalog()`, `BuiltinContextPlugin` with 12 stub providers, `intergrax.context` entry point, quality module move (CE-1.5), and Tier-3 wiring with plugin id validation.

## Project impact

Hosts bootstrap context plugins alongside integrations/tools/skills; authors can register third-party `ContextPlugin` classes via entry points.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CONTEXT_ENGINEERING.md` §8.4 |
| Plan | `docs/plan/CONTEXT_ENGINEERING.md` — S2 Done; CE-1.5, CE-2.1–CE-2.5 |
| ADR | `docs/adr/entries/2026-06-12/ADR-CTX-001.md` (no new ADR) |

## Changed artifacts

- `intergrax/context/providers/builtin.py` — `BuiltinContextPlugin` + stub providers
- `intergrax/context/catalog.py` — `bootstrap_context_catalog()`
- `intergrax/context/quality.py` — moved from runtime (CE-1.5)
- `applications/_shared/context_wiring.py` — plugin id validation
- `tests/unit/context/test_context_catalog.py` — catalog bootstrap gate tests

## Verification

```bash
uv run pytest tests/unit/context/ tests/unit/applications/test_context_plugin_wiring.py -m gate -q
python scripts/check_context_tier0_import_boundary.py
```

Result: 19 passed (context suite).

## Risks and follow-ups

- Builtin provider `collect()` stubs are catalog placeholders; live collect ships in CE-7 (workspace) and CE-VEC-1 (session semantic).
- Sprint S3 wires `DefaultNexusContextEngine` on the compiler hot path.
