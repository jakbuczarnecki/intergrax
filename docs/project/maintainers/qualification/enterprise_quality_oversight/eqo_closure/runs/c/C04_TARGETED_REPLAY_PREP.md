# EQO-C-04 — targeted replay prep (wave-2)

**Parent:** `EQO-C-04` · **Prereq:** `EQO_C_DRIFT.jsonl` / C-03 `TARGETED_REPLAY` rows

## Status

`DEFERRED` — wave-2 does not execute band-C targeted replay (pytest lock held for audit prep + prior EXE manifests). Wave-3 or dedicated C lane may claim lock after operator queue.

## Prep checklist (no run)

- [ ] Enumerate `TARGETED_REPLAY` rows from C-03 drift register
- [ ] Pin worktree `eqo-w3` @ TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
- [ ] Acquire `D:/Projekty/_wt/_locks/pytest.lock`
- [ ] Store logs under `eqo_closure/runs/c/`

**Ref:** `governance/tickets/T-IGX-046-F.md`