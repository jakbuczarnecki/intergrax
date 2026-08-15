---
id: IJ-2026-06-11-011
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-CI-1
  - ACP-CLOSE-CI-3
status: completed
commit: pending
adr: none — wires existing fleet/readiness scripts into regression gate workflow
---

# ACP-CLOSE CI-1/CI-3 — gate workflow fleet + scoreboard blockers

## Operator request

Execute next ACP-CLOSE sprint: wire post-LEG fleet migration grep and production readiness `--fail-on-blockers` into CI gate workflow.

## Summary

Added `check_agent_acp_close_ci.py` aggregate gate (fleet inventory audit, `check_agent_fleet_migration.py`, production readiness with `--fail-on-blockers` + fleet closure + mutating 100%). Wired into `.github/workflows/unit-tests.yml` (fast + tier governance), `check_agent_release_gates.py`, and ACP CI conformance matrix rows CI-04 and CI-16.

## Project impact

Regression gate now blocks Tier-2 `RuntimeEngine` reintroduction and scoreboard dimension blockers on every PR. ACP-CLOSE CI band P1 items closed.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.10 CI-04 · §40.15 CI-16 |
| Plan | `ACP-CLOSE-CI-1`, `ACP-CLOSE-CI-3` |
| ADR | none |

## Changed artifacts

- `scripts/gates/check_agent_acp_close_ci.py` — aggregate ACP-CLOSE CI gate (new)
- `.github/workflows/unit-tests.yml` — invoke in fast-gate and gate-governance-tier
- `scripts/gates/check_agent_release_gates.py` — consume aggregate gate
- `scripts/gates/check_acp_ci_conformance_matrix.py` — CI-04 + CI-16 rows
- `scripts/maintenance/check_agent_threat_model.py` — threat gate reference
- `tests/unit/scripts/test_check_agent_acp_close_ci.py` — gate self-test (new)
- `AGENTS.md` — verification command

## Verification

```bash
uv run python scripts/gates/check_agent_acp_close_ci.py
uv run pytest tests/unit/scripts/test_check_agent_acp_close_ci.py -m gate -q
```

Result: pass.

## Risks and follow-ups

- ACP-CLOSE-CI-2 (anti-pattern ACP-AP-02 after TOOL-ENG-6) remains open.
- Fresh CI clones need no prebuilt `build/` artifacts — aggregate gate regenerates inventory.
