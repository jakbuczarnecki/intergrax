# EQO-W4-02 — DOC_FIX tranche 01 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 1 of N  
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).  
**Cap:** 15 IDs (first eligible in file order).  
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A1-001 | MEDIUM | Roadmap owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A1-002 | MEDIUM | Roadmap owner (Operator) / PROD-Q | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A1-003 | MEDIUM | TRACE-X / PROD-Q owners | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A1-004 | MEDIUM | Roadmap owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A1-005 | LOW | Roadmap owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 6 | EQOA-A1-006 | MEDIUM | Roadmap owner (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 7 | EQOA-A1-009 | MEDIUM | HARNESS residual owner (Harness AI program) / ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 8 | EQOA-A1-010 | MEDIUM | ARCH-FREEZE / TRACE-X SSOT owner (SIGN-6_TRACE publication) | Doc fix in owner ticket or `BLOCKED` |
| 9 | EQOA-A1-013 | MEDIUM | Roadmap S01 maintainers (PROD-Q stage owner) | Doc fix in owner ticket or `BLOCKED` |
| 10 | EQOA-A1-014 | MEDIUM | PROD-Q stage owner (roadmap S01) | Doc fix in owner ticket or `BLOCKED` |
| 11 | EQOA-A1-019 | LOW | Roadmap maintainers (EBH-6 stage owner) | Doc fix in owner ticket or `BLOCKED` |
| 12 | EQOA-A1-020 | MEDIUM | PDC-X program owner | Doc fix in owner ticket or `BLOCKED` |
| 13 | EQOA-A1-022 | INFO | Roadmap maintainers | Doc fix in owner ticket or `BLOCKED` |
| 14 | EQOA-A1-023 | MEDIUM | Autonomous Work / Integrations (AW-7C) | Doc fix in owner ticket or `BLOCKED` |
| 15 | EQOA-A1-024 | LOW | Governance domain | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- Remaining `DOC_FIX_BY_OWNER` rows outside Campaign B: continue in tranche 02+ using the same exclusion rule.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **S3** (`eqo/par/s3-w4x`); integrate via `eqo/closure-v2-phase-0-9` ladder.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.