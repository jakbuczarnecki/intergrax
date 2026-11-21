# EQO wave 10 — W3 EXE rerun manifest (WB01,06,07,10,11)

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `eqo_closure/runs/wb01|wb06|wb07|wb10|wb11/` (logs, `RUN_MANIFEST.json`, disposition touch-ups), this manifest, README index row.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `6b01b63f6264ec62602ba6349d532686dbf8c54c` (post PR #707 wave-9 integ).

**Pytest env:** `eqo-w3` @ TRAIN_PIN; `uv sync --all-extras` (includes `llm-langchain-ollama`); shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Lanes

| Lane | Branch | Worktree | Workstreams |
|------|--------|----------|-------------|
| **G-EXE** | `eqo/wave-10/exe` | `eqo-w10-exe` | WB01, WB06, WB07, WB10, WB11 pytest EXE @ TRAIN_PIN |
| **G-INT** | `eqo/wave-10/integ` | `eqo-w10-integ` | Lane merge, manifest, README index, PR → `development` |

**Out of scope:** product/FRZ edits, AUD session E verdict changes, terminal closure claims, G6/IIAC independence claims.

## EXE summary (@ TRAIN_PIN)

| WB | DES prereg | `pytest_exit_code` | Summary | Blockers |
|----|------------|-------------------:|---------|----------|
| WB01 | — (manifest-only) | `1` | 1 failed, 23 passed, 1 skipped in 314.65s | `PR680_NOT_ON_TRAIN_PIN`, `IIAC_NOT_IN_FORCE` |
| WB06 | yes | `0` | 32 passed in 188.25s | — |
| WB07 | yes | `1` | 1 failed, 56 passed, 1 skipped in 107.04s | `PYTEST_SLICE_NON_ZERO_EXIT` |
| WB10 | yes | `1` | 6 failed, 66 passed, 1 skipped in 98.73s | `PYTEST_SLICE_NON_ZERO_EXIT` |
| WB11 | yes | `1` | 1 failed, 68 passed in 140.29s (0:02:20) | `PYTEST_SLICE_NON_ZERO_EXIT` |

Evidence paths: `runs/wbNN/pytest_wbNN_exe.log`, `runs/wbNN/RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-10/exe`).

## Merge ladder

```text
eqo/wave-10/exe (WB01/06/07/10/11) ──► eqo/wave-10/integ ──► development (PR; docs gates)
```

**Docs gates (integrator PR):** `ticket-governance` + `governance-policy-pin` per T-IGX-046-F merge policy.
