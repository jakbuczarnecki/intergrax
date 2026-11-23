# EQO-W4-02 — DOC_FIX tranche 05 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 5 of N
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).
**Cap:** 15 IDs (eligible in file order after tranche 04).
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A4-041 | LOW | ARCH-FREEZE documentation; QUAL-X owner | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A5-004 | LOW | Harness / TOOL-LIFE-X owner | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A5-005 | INFO | Tools domain owner | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A5-008 | MEDIUM | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A5-014 | LOW | PRE-ENT campaign owner (T-IGX-034-A) / Operator | Doc fix in owner ticket or `BLOCKED` |
| 6 | EQOA-A6-003 | LOW | ENT-AUDIT-X | Doc fix in owner ticket or `BLOCKED` |
| 7 | EQOA-A6-005 | MEDIUM | ENT-AUDIT-X | Doc fix in owner ticket or `BLOCKED` |
| 8 | EQOA-A6-008 | LOW | ENT-AUDIT-X | Doc fix in owner ticket or `BLOCKED` |
| 9 | EQOA-A7-001 | MEDIUM | ENT-AUDIT-X | Doc fix in owner ticket or `BLOCKED` |
| 10 | EQOA-A7-002 | LOW | ARCH-FREEZE / Observability / platform | Doc fix in owner ticket or `BLOCKED` |
| 11 | EQOA-A7-004 | LOW | platform | Doc fix in owner ticket or `BLOCKED` |
| 12 | EQOA-A7-006 | MEDIUM | EBH-6 owner | Doc fix in owner ticket or `BLOCKED` |
| 13 | EQOA-A7-011 | LOW | DUP-X / EBH-6 owners | Doc fix in owner ticket or `BLOCKED` |
| 14 | EQOA-A7-012 | MEDIUM | PDC-X | Doc fix in owner ticket or `BLOCKED` |
| 15 | EQOA-A7-016 | LOW | PDC-X D1/D2/D6/D7/D8 | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- Remaining `DOC_FIX_BY_OWNER` rows outside Campaign B: continue in tranche 06+ using the same exclusion rule.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **W4** (`eqo/wave-2/w4`); integrate via `eqo/wave-2/integ`.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.
