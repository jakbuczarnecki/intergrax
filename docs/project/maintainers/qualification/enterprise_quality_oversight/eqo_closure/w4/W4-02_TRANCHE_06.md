# EQO-W4-02 — DOC_FIX tranche 06 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 6 of N
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).
**Cap:** 15 IDs (eligible in file order after tranche 05).
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A7-020 | MEDIUM | TRACE-X / roadmap owner | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A7-022 | LOW | MCP-SERVER-X | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A7-024 | LOW | HIST campaign / roadmap owner | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A7-026 | MEDIUM | Operator / ticket owners | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A7-032 | LOW | EBH-6 / DUP-X | Doc fix in owner ticket or `BLOCKED` |
| 6 | EQOA-A7-033 | LOW | Register owners (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 7 | EQOA-A7-034 | MEDIUM | Register owners (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 8 | EQOA-A7-035 | MEDIUM | Register owners (Operator) | Doc fix in owner ticket or `BLOCKED` |
| 9 | EQOA-A8-001 | LOW | PDC-X D5 (reconciliation PDC-D5-R016) | Doc fix in owner ticket or `BLOCKED` |
| 10 | EQOA-A8-005 | LOW | PDC-X D1/D2 and domain owners | Doc fix in owner ticket or `BLOCKED` |
| 11 | EQOA-A8-007 | LOW | Domain owners | Doc fix in owner ticket or `BLOCKED` |
| 12 | EQOA-A8-008 | MEDIUM | Stage owner (AW / Capability Catalog) | Doc fix in owner ticket or `BLOCKED` |
| 13 | EQOA-A8-009 | MEDIUM | AW stage owner | Doc fix in owner ticket or `BLOCKED` |
| 14 | EQOA-A8-010 | MEDIUM | ERL stage owner / C17 | Doc fix in owner ticket or `BLOCKED` |
| 15 | EQOA-A8-011 | LOW | COMPAT-X owner | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- Remaining `DOC_FIX_BY_OWNER` rows outside Campaign B: continue in tranche 07+ using the same exclusion rule.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **W4** (`eqo/wave-2/w4`); integrate via `eqo/wave-2/integ`.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.
