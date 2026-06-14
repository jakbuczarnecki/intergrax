---
id: IJ-2026-06-14-005
date: 2026-06-14
tiers:
  - tier-0
  - tier-2
scope: SKILLS
plan_ref:
  - SK-EXP5
  - ACP-CLOSE-CI
status: completed
commit: pending
adr: none — test/doc/fixture alignment; no contract or architecture change
---

# Gate regression fix — skill catalog drift, fleet roster, pytest gate scope

## Operator request

Fix unacceptable full-gate runtime (~12 min) and all failing gate tests after MEM-VEC iteration; align verification commands with CI (`gate and not no_ci`).

## Summary

- Updated shipped skill catalog expectations to **150 skills / 42 bundles** after `codecraft.ephemeral_builder` (ECC-2.7).
- Added `codecraft` bundle and per-skill `USAGE.md` documentation.
- Registered `boundary_demo` as migrated in `audit_agent_fleet_legacy.py` so §40.12 readiness and fleet scoreboard gates pass.
- `tests/conftest.py`: deselect `no_ci` tests when `-m gate` is used without explicit `no_ci` (751 s → ~49 s locally).
- Session autouse fixture regenerates `build/agent_fleet_inventory.json` on each pytest session.
- `AGENTS.md` and `.cursor/rules/intergrax-iteration.mdc` now document `pytest -m "gate and not no_ci"`.

## Project impact

Local and agent verification matches nightly CI gate profile; accidental full-repo script subprocess suites no longer run under bare `-m gate`. Skill catalog and readiness gates green again.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/SKILLS.md` |
| Plan | `docs/plan/SKILLS.md` · `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.12 |
| ADR | none — fixture/doc/test alignment only |

## Changed artifacts

- `intergrax/skills/providers/codecraft/USAGE.md` — bundle index
- `intergrax/skills/providers/codecraft/codecraft.ephemeral_builder/USAGE.md` — skill guide
- `scripts/audit_agent_fleet_legacy.py` — `boundary_demo` migrated roster
- `tests/conftest.py` — gate/no_ci deselection + fleet inventory refresh
- `tests/unit/skills/test_*.py` — catalog count 150 / 42 bundles
- `AGENTS.md` · `.cursor/rules/intergrax-iteration.mdc` — verification command

## Verification

```bash
uv run pytest -m gate -q
uv run pytest -m "gate and not no_ci" -q
python scripts/check_implementation_journal.py
```

Result: **1390 passed** in ~49 s (both gate invocations); journal check OK.

## Risks and follow-ups

- Explicit deep audit still available via `-m no_ci` or nightly governance CI jobs (direct script invocation).
- `build/agent_fleet_inventory.json` remains gitignored; session fixture regenerates it.
