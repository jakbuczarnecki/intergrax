# EQO wave 13 — remaining W3 EXE + doc disposition manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `eqo_closure/runs/wb02|wb03|wb05|wb08|wb12/` (logs, `RUN_MANIFEST.json`), EDER/C-04/W4-03 doc lanes, this manifest, README index row, matrix EXE refresh.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `ad90940363e43a3273d4965cef87dd86d9439a4f`.

**Pytest env:** `eqo-w3` @ TRAIN_PIN; `uv sync --all-extras`; shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Parallel lanes

| Lane | Branch | Worktree | Deliverable | Commit |
|------|--------|----------|-------------|--------|
| **G-EXE-A** | `eqo/wave-13/exe-wb02`, `exe-wb03`, `exe-wb05` | `eqo-w13-exe-NN` | pytest logs + manifests | `d1c1f4128`, `3fb15ba6c`, `e19f64e05` |
| **G-EXE-B** | `eqo/wave-13/exe-wb08`, `exe-wb12` | `eqo-w13-exe-08`, `exe-12` | WB08 first EXE + WB12 refresh | `496c5169a`, `125b343c7` |
| **G-E** | `eqo/wave-13/e` | `eqo-w13-e` | `x/N-0001|N-0002/EDER_E03_ROUTE.md` | `07e4d0d13` |
| **G-C04** | `eqo/wave-13/c04` | `eqo-w13-c04` | `runs/c/C04_STATUS.md` cross-ref BLOCKED | `344ce5e38` |
| **G-W4** | `eqo/wave-13/w4` | `eqo-w13-w4` | `W4-03_OWNER_BATCH_ENGLISH_MEMO.md` | `8271d1459` |
| **G-INT** | `eqo/wave-13/integ` | `eqo-w13-integ` | This manifest, README, matrix, PR → `development` | _(integrator tip)_ |

**Out of scope:** `EQO_CLOSED`, Session E verdict changes, EDER register apply, C-04 pytest re-run, wave-14 CLS terminal band.

## EXE summary (@ TRAIN_PIN, wave-13)

| WB | `pytest_exit_code` | Summary | Notes |
|----|-------------------:|---------|-------|
| WB02 | `1` | 2 failed, 72 passed, 1 skipped in 17.77s | Non-zero recorded; no closure claim |
| WB03 | `0` | 39 passed in 99.47s | — |
| WB05 | `0` | 14 passed in 32.86s | — |
| WB08 | `0` | 44 passed in 7.33s | First EXE log @ pin |
| WB12 | `0` | 33 passed, 1 skipped in 48.97s | — |

Evidence: `runs/wbNN/pytest_wbNN_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-13/exe-wbNN`).

## Merge ladder

```text
exe-wb02,03,05,08,12 ──► G-E + G-C04 + G-W4 ──► eqo/wave-13/integ ──► development (PR; docs gates)
```

**Docs gates:** `ticket-governance` + `governance-policy-pin` per T-IGX-046-F waiver pattern.

**Follow-on:** Wave 14 CLS terminal band waits on this merge + fresh EXE logs (operator approved).