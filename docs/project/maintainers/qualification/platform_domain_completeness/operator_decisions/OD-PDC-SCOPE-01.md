# OD-PDC-SCOPE-01 — Skill disable/remove/upgrade (SKILL-X F-04) — implement or authorize exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-01 |
| **Stage** | PDC-X |
| **Status** | **PENDING** |
| **Requirements** | PDC-D1-R018 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

SKILL-X closed PASS_SCOPED with skill disable, remove and upgrade unsupported and untested (`skills/host_lifecycle.py` has only `enable_installed_skill` and `bind`). Is this mandatory before ARCH-FREEZE?

## Recommended option (A)

**Implement** disable/remove/upgrade on `SkillHostLifecycleService`, extending the Skills manifest model (GAP_AUDIT §16 PRESERVE/EXTEND), qualified in `tests/qualification/pdc_x/d1/`. Recommended: a skill that cannot be removed or upgraded leaves agent upgrade/retire (ADR-PDC-01) unable to complete for skill-bearing agents.

## Alternate option (B)

Authorized post-freeze exclusion, recorded as a freeze limitation; agent lifecycle certification must then exclude skill-bearing upgrade paths.

## Operator record

_Pending._
