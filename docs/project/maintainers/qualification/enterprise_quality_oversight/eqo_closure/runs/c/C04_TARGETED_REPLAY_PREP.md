# EQO-C-04 — targeted replay prep (wave-2)

**Parent:** `EQO-C-04` · **Prereq:** `EQO_C_DRIFT.jsonl` / C-03 `TARGETED_REPLAY` rows

## Status

`BLOCKED` — wave-3 executed targeted replay @ TRAIN_PIN (lock acquired); pytest exit **1** (`47 failed, 395 passed` per log). Wave-7: log-derived status in `C04_STATUS.md`; manifest refresh in `C04_RUN_MANIFEST.json`. No re-run and no product fix in this ticket.

## Prep checklist

- [x] Enumerate `TARGETED_REPLAY` rows: TRACE-X, MCP-SERVER-X, CODEOWNERS_DELTA (`EQO_C_DRIFT.jsonl`)
- [x] Pin worktree `eqo-w3` @ TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
- [x] Acquire `D:/Projekty/_wt/_locks/pytest.lock`
- [x] Store logs under `eqo_closure/runs/c/`

**Ref:** `governance/tickets/T-IGX-046-F.md`