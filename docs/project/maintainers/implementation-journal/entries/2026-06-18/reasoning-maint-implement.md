---
id: IJ-2026-06-18-036
date: 2026-06-18
tiers:
  - tier-1
scope: REASONING_AND_COGNITION
plan_ref:
  - COG-MAINT-03
status: completed
commit: pending
adr: none — acceptance test only
---

# COG-MAINT-03 — audit maintenance implementation

## Summary

Added acceptance test proving `allow_dynamic_replan` allows MODIFY_PLAN after policy interrupt on `ApplicationEnvironmentProfile.lab_defaults()` reference host wiring.

## Changed artifacts

- `tests/acceptance/agent_os/test_cog_maint_replan.py`
- `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` §6.1av

## Verification

```bash
uv run pytest tests/acceptance/agent_os/test_cog_maint_replan.py -q
```

## Risks

None — exercises existing interrupt handler contract.
