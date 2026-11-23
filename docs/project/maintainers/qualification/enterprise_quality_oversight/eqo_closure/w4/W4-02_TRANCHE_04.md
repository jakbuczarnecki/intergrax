# EQO-W4-02 — DOC_FIX tranche 04 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 4 of N
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).
**Cap:** 15 IDs (eligible in file order after tranche 03).
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A3-023 | LOW | QUAL-X / TOOL-LIFE-X stage owners | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A3-024 | LOW | CTRL-X / QUAL-X / PDC-X D6 | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A3-028 | LOW | ARCH-FREEZE B7 (T-B7-PROMOTE-BND) | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A3-029 | LOW | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A4-005 | MEDIUM | Dependency governance owner (DEP4) | Doc fix in owner ticket or `BLOCKED` |
| 6 | EQOA-A4-010 | MEDIUM | Governance domain owner; EBH-6 / FRZ-DOC rows | Doc fix in owner ticket or `BLOCKED` |
| 7 | EQOA-A4-014 | LOW | Agent contracts domain owner | Doc fix in owner ticket or `BLOCKED` |
| 8 | EQOA-A4-018 | MEDIUM | LLM domain owner (EBH-2E) / Modality owner | Doc fix in owner ticket or `BLOCKED` |
| 9 | EQOA-A4-021 | LOW | MCP-SERVER-X owner | Doc fix in owner ticket or `BLOCKED` |
| 10 | EQOA-A4-022 | LOW | Tools domain owner | Doc fix in owner ticket or `BLOCKED` |
| 11 | EQOA-A4-023 | LOW | Tools / Marketplace / Platform Plugins owners | Doc fix in owner ticket or `BLOCKED` |
| 12 | EQOA-A4-028 | LOW | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 13 | EQOA-A4-029 | LOW | Memory domain owner | Doc fix in owner ticket or `BLOCKED` |
| 14 | EQOA-A4-033 | MEDIUM | ARCH-FREEZE | Doc fix in owner ticket or `BLOCKED` |
| 15 | EQOA-A4-038 | LOW | EVOL-X / ENT-AUDIT-X / PDC-X owners (ARCH-FREEZE documentation) | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- Remaining `DOC_FIX_BY_OWNER` rows outside Campaign B: continue in tranche 05+ using the same exclusion rule.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **S3** (`eqo/par/s3-w4x`); integrate via `eqo/closure-v2-phase-0-9` ladder.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.
