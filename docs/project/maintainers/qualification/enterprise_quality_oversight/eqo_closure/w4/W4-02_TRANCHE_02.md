# EQO-W4-02 — DOC_FIX tranche 02 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 2 of N
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).
**Cap:** 15 IDs (eligible in file order after tranche 01).
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A2-006 | LOW | IIAC-GOV-1 / EVOL-X owners | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A2-007 | MEDIUM | Stage owners / ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A2-010 | MEDIUM | Autonomous Work / Integrations | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A2-011 | LOW | Roadmap owner | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A2-012 | LOW | Integrations domain | Doc fix in owner ticket or `BLOCKED` |
| 6 | EQOA-A2-014 | MEDIUM | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 7 | EQOA-A2-015 | MEDIUM | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 8 | EQOA-A2-016 | MEDIUM | PDC-X | Doc fix in owner ticket or `BLOCKED` |
| 9 | EQOA-A2-018 | LOW | SCENARIO-FACTORY-X | Doc fix in owner ticket or `BLOCKED` |
| 10 | EQOA-A2-021 | LOW | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 11 | EQOA-A2-026 | MEDIUM | QUAL-X / SKILL-X / TOOL-LIFE-X owners | Doc fix in owner ticket or `BLOCKED` |
| 12 | EQOA-A2-028 | LOW | TOOL-LIFE-X | Doc fix in owner ticket or `BLOCKED` |
| 13 | EQOA-A2-029 | MEDIUM | EBH-6 stage owner / Operator | Doc fix in owner ticket or `BLOCKED` |
| 14 | EQOA-A2-032 | MEDIUM | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 15 | EQOA-A2-033 | LOW | Stage owners | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- Remaining `DOC_FIX_BY_OWNER` rows outside Campaign B: continue in tranche 03+ using the same exclusion rule.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **S3** (`eqo/par/s3-w4x`); integrate via `eqo/closure-v2-phase-0-9` ladder.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.
