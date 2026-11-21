# EQO-W4-02 — DOC_FIX tranche 07 (outside Campaign B)

**Parent:** EQO-W4-02 · **Batch:** 7 of N
**Selection rule:** rows in `EQO_B2_DISPOSITION.jsonl` with `next_action_source=DOC_FIX_BY_OWNER` whose `finding_id` is **not** among the 51 Campaign B-1 IDs (`EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`).
**Cap:** 15 IDs (eligible in file order after tranche 06).
**Scope of this artefact:** planning queue only — no doc PR, no matrix edit, no roadmap/FRZ change.

## Tranche queue

| # | Finding ID | Severity | Owner (disposition) | Next action |
|---|------------|----------|---------------------|-------------|
| 1 | EQOA-A8-015 | LOW | Domain owners | Doc fix in owner ticket or `BLOCKED` |
| 2 | EQOA-A8-016 | LOW | PLATFORM_PLUGINS owner | Doc fix in owner ticket or `BLOCKED` |
| 3 | EQOA-A8-018 | MEDIUM | EVOL-X / MCP-SERVER-X / SFX / PDC-X doc owners | Doc fix in owner ticket or `BLOCKED` |
| 4 | EQOA-A8-019 | LOW | Docs governance / feature owners | Doc fix in owner ticket or `BLOCKED` |
| 5 | EQOA-A8-023 | LOW | Feature doc owners | Doc fix in owner ticket or `BLOCKED` |

## Steward notes

- **W4-02 queue complete** for eligible DOC_FIX rows outside Campaign B.
- Executor does not assign persons; missing named ticket → row stays `BLOCKED` per `EQO_B2_REMAINING_INVARIANTS_PLAN.md`.
- Parallel group **W4** (`eqo/wave-2/w4`); integrate via `eqo/wave-2/integ`.

**Recorded:** 2026-11-21 · pins per `EQO_PIN_BIND.json`.
