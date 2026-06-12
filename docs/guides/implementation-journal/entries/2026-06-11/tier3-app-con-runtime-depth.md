---
id: IJ-2026-06-11-020
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - H-APP-CON-DOC.1
  - APP-CON-2
  - APP-CON-4
  - APP-PROD-1
status: completed
commit: pending
adr: none — documentation + contract expansion; kernel budget enforcement remains ACP-TOK-2
---

# TIER3 APP-CON runtime depth — state v2, artifacts, hooks, prod gates

## Operator request

Deepen Tier-3 application environment architecture from strong (9/10) to top-tier by closing runtime detail gaps: budget governance flow, richer `ApplicationEnvironmentState`, hook determinism contract, shadow/sandbox lifecycle, artifact model, APP-PROD gates, and Developer Mental Model — without introducing Tier-3 cognition loop.

## Summary

Expanded `TIER3_APPLICATION_ENVIRONMENT.md` with §32.6 hook runtime contract, §42 v2 state model, §43 full budget flow, §47 Developer Mental Model, §48 Application Artifacts; updated §20–§21 lifecycle, §40 APP-PROD register, §46 maturity score. Implemented `ApplicationEnvironmentState` v2, `application_artifacts.py`, `check_application_production_gates.py`, and unit tests.

## Project impact

Tier-3 hosts now have normative contracts for host state, artifacts, hook ordering, and production gate script; developers get recipe-oriented mental model. Budget kernel enforcement (ACP-TOK-2/3) remains explicitly planned — architecture documents honest status.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §32.6, §42–§43, §47–§48 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` Phase H-APP-CON |
| ADR | none — ACP-TOK-2 will need harness ADR when implemented |

## Changed artifacts

- `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — runtime depth sections
- `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — H-APP-CON register update
- `intergrax/applications/contracts/environment_state.py` — v2 models
- `intergrax/applications/contracts/application_artifacts.py` — artifact refs
- `scripts/check_application_production_gates.py` — APP-PROD-1
- `tests/unit/applications/test_environment_state_and_artifacts.py` — contract tests

## Verification

```bash
uv run pytest tests/unit/applications/test_environment_state_and_artifacts.py tests/unit/applications/test_application_host_wiring.py -q
python scripts/check_application_production_gates.py
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
```

Result: pass (expected).

## Risks and follow-ups

- ACP-TOK-2/3 kernel budget enforcement + host notify — blocks mutating STRICT production claim.
- APP-CON-3 auto-seed/sync state on Nexus lifecycle hooks.
- APP-PROD-6..8 CI gates (environment_state usage, budget, workspace cleanup).
- Wire `check_application_production_gates.py` into gate workflow.
