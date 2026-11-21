# EQO-W3-WB03-DES — scope notes (Campaign B-1)

**Source:** `EQO_B1_CRITICAL_INVARIANTS_PLAN.md` · **Pin:** `TRAIN_PIN` (`28c27cb`) · **Parallel group:** S1 · **Milestone:** B (`DES` only; no `EXE` in this batch).

| Field | Value |
|-------|-------|
| Workstream | WB03 |
| Domains | Fail-closed, HITL |
| Finding IDs | EQOA-A4-026, EQOA-A4-031, EQOA-A4-037, EQOA-A3-027 |
| Ticket refs | T-IGX-043-A, T-IGX-043-B, T-IGX-043-E |
| Primary matrix | `EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl` |
| Write set | `eqo_closure/runs/wb03/` only (no product tree) |

**Merge dependency (#680):** findings A4-037 (with A4-044, A4-015) may need probe before merge on `TRAIN_PIN`; after pin move → `EQO-W3-WB03-DRV`, not a new campaign.

**Acceptance (DES):** preregistered SHA256 of the band-C pytest command plus hypotheses on `TRAIN_PIN` (`DES_PREREG.json`).