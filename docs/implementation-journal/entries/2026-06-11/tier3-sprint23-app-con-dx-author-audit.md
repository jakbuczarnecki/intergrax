---
id: IJ-2026-06-11-048
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-DX.1
  - APP-CON-DX.2
status: completed
commit: pending
adr: none — author guide + audit prompt refresh; no runtime changes
---

# Sprint 23 — Application author guide + Tier-3 audit prompt (APP-CON-DX)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-CON-DX.1 and APP-CON-DX.2 — author mental model/checklist guide and regenerated domain audit prompt for architecture §24–§51.

## Summary

- `APPLICATION_CREATION_GUIDE.md` — canonical §31 workflow, §45 checklist, §47 recipes, ops CLI, verification commands.
- `AGENT_CREATION_GUIDE.md` Appendix F links to application guide.
- `generate_domain_audit_prompts.py` — TIER3 domain expanded for APP-CON/PROD/EVOL/OPS gates and §24–§51 scope.
- Regenerated `docs/audit/TIER3_APPLICATION_ENVIRONMENT.md`.
- `scripts/check_tier3_audit_prompt.py` wired into production gates.

## Project impact

Application authors have a single creation guide aligned with frozen architecture canon; audit agents receive an up-to-date Tier-3 prompt covering evolution, ops, and production gates. Closes §6.2y post-freeze backlog.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §31 · §45 · §47 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-CON-DX.1 · APP-CON-DX.2 · §6.2y step 15 |

## Changed artifacts

- `docs/guides/APPLICATION_CREATION_GUIDE.md`
- `docs/guides/AGENT_CREATION_GUIDE.md`
- `docs/audit/TIER3_APPLICATION_ENVIRONMENT.md`
- `scripts/generate_domain_audit_prompts.py`
- `scripts/check_tier3_audit_prompt.py`
- `scripts/check_application_production_gates.py`
- `AGENTS.md`
- `tests/unit/guides/test_application_creation_guide.py`
- `tests/unit/scripts/test_check_tier3_audit_prompt.py`

## Verification

```bash
uv run pytest tests/unit/guides/test_application_creation_guide.py \
  tests/unit/scripts/test_check_tier3_audit_prompt.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/check_tier3_audit_prompt.py
python scripts/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- §6.2y APP backlog complete — remaining Tier-3 work is cross-plan (ACP-TOK) or product-specific, not structural canon.
- Regenerating all audit domains touches 22 files when generator runs — commit only Tier-3 delta or full regen per operator preference.
