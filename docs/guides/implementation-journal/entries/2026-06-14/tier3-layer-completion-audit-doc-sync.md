---
id: IJ-2026-06-14-010
date: 2026-06-14
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - T3-LC-01
  - T3-LC-02
  - T3-LC-03
  - LC-DOC
  - LC-IMPL-1
status: completed
commit: dfed30ba
adr: no ADR needed — documentation sync and missing import fix; no contract change
---

# Tier-3 Layer Completion — doc sync + runtime_config_bridge import fix

## Operator request

Run Tier-3 Layer Completion Mode: audit APP-EVOL/OPS/recovery/package/registry areas flagged as still "planned" in hub review, sync documentation with implementation, close P0/P1 gaps, and deliver maturity assessment.

## Summary

Layer completion audit confirmed APP-EVOL-1..7 and APP-OPS-1..4 are **Done** in code and gates; the perceived "planned" state came from stale `(target)` section headers in architecture §49–§50 and governance audit rows. Synced headers, hub, readiness checklist, and plan register. Fixed P1 `NameError` in `runtime_config_bridge.py` (missing `derive_run_budget_from_context_policy` import from LLM layer work).

## Project impact

Tier-3 documentation now matches implementation status: **Architecturally Mature** for reference hosts. Operators reading §49–§50 no longer see false "planned" signals. `materialize_runtime_config` again derives `RunBudget` from context policy when unset (10 previously failing unit tests restored).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §46.3 · §49–§50 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` Tier-3 Layer Completion Audit |
| Governance | `docs/guides/GOVERNANCE_CONSISTENCY_AUDIT.md` §2 |
| Hub | `docs/intergrax_runtime_architecture.md` Application section |

## Changed artifacts

- `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — `(target)` → **Done** on APP-EVOL/OPS sections; maturity table 2026-06-14
- `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — layer completion audit register + P3/P4 backlog
- `docs/guides/GOVERNANCE_CONSISTENCY_AUDIT.md` — CapabilityAlias **Done**
- `docs/intergrax_runtime_architecture.md` — APP-EVOL/OPS maturity line
- `applications/TIER3_READINESS.md` — platform maturity note
- `intergrax/applications/_shared/runtime_config_bridge.py` — import fix

## Verification

```bash
uv run pytest tests/unit/applications/ -q
uv run python scripts/check_application_production_gates.py
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
```

Result: 453 passed; all gates OK.

## Risks and follow-ups

- P4: `graph_version` / `envelope_version` on graph/envelope models (migration schema only today)
- P4: marketplace UI and signed package distribution channel
- P3: queue worker scaffold-default (opt-in by design)
