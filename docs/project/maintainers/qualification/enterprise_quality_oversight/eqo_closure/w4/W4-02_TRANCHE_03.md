# EQO-W4-02 — DOC_FIX tranche 03 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 3 of N
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).
**Cap:** 15 IDs (eligible in file order after tranche 02).
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A2-036 | MEDIUM | ENT-AUDIT-X | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A2-039 | LOW | Roadmap owner | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A2-044 | LOW | HARNESS program | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A2-045 | MEDIUM | Harness program owner | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A2-046 | MEDIUM | Harness program owner | Doc fix in owner ticket or `BLOCKED` |
| 6 | EQOA-A3-001 | MEDIUM | FRZ checklist owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 7 | EQOA-A3-002 | MEDIUM | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 8 | EQOA-A3-003 | LOW | PDC-X | Doc fix in owner ticket or `BLOCKED` |
| 9 | EQOA-A3-005 | LOW | FRZ checklist owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 10 | EQOA-A3-006 | LOW | FRZ checklist owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 11 | EQOA-A3-009 | LOW | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 12 | EQOA-A3-015 | LOW | Operator | Doc fix in owner ticket or `BLOCKED` |
| 13 | EQOA-A3-017 | MEDIUM | HARNESS-FINAL owner / ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 14 | EQOA-A3-018 | LOW | STATE-X owner | Doc fix in owner ticket or `BLOCKED` |
| 15 | EQOA-A3-020 | LOW | Checklist maintainer | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- Remaining `DOC_FIX_BY_OWNER` rows outside Campaign B: continue in tranche 04+ using the same exclusion rule.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **S3** (`eqo/par/s3-w4x`); integrate via `eqo/closure-v2-phase-0-9` ladder.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.
