# BATCH_SETTLEMENT — EC-W21-W3-DETERMINISTIC-EXECUTION-02

**PRE_G2_DIAGNOSTIC_REPLAY — NOT FINAL QUALIFICATION**

Execution SHA: `786088a9798ec18a26c22778f8217608e5f15bff`  
Historical evidence SHA: `28c27cb896d02e0185b9fa02047411fa152f6469`

## Package A
- A-01: EQOA-A1-021 workstream WB11→WB01 in detailed finding table.
- A-02: Fixture classification clarification appended to EQO_W21_CI_DISPOSITION.md.

## Package B — historical vs current

| WB | Hist P/F/S | Exit | Current summary | Classification |
|---|---|---|---|
| WB01 | 1 failed, 23 passed, 1 skipped in 314.65s (0:05:14) | 1 | 1 failed, 23 passed, 1 skipped in 119.13s (0:01:59) | REPLAY_RED |
| WB02 | 2 failed, 72 passed, 1 skipped in 17.77s | 1 | 66 passed, 1 skipped in 24.17s | REPLAY_GREEN_WITH_LIMITS |
| WB03 | 39 passed in 99.47s (0:01:39) | 0 | 39 passed, 1 warning in 66.58s (0:01:06) | REPLAY_GREEN |
| WB04 | 13 passed in 79.41s | 0 | 1 failed, 12 passed, 1 warning in 19.38s | REPLAY_RED |
| WB05 | 14 passed in 32.86s | 0 | 1 failed, 19 passed in 24.88s | REPLAY_RED |
| WB06 | 32 passed in 188.25s (0:03:08) | 0 | 1 error in 0.36s | EXECUTION_BLOCKED |
| WB07 | 1 failed, 56 passed, 1 skipped in 107.04s (0:01:47) | 1 | 2 failed, 55 passed, 1 skipped in 53.04s | REPLAY_RED |
| WB08 | 44 passed in 7.33s | 0 | 44 passed in 5.66s | REPLAY_GREEN |
| WB09 | 66 passed in 66.86s | 0 | 2 failed, 64 passed, 2 warnings in 9.04s | REPLAY_RED |
| WB10 | 6 failed, 66 passed, 1 skipped in 98.73s (0:01:38) | 1 | 3 failed, 69 passed, 1 skipped in 47.11s | REPLAY_RED |
| WB11 | 1 failed, 68 passed in 140.29s (0:02:20) | 1 | 1 failed, 68 passed in 106.95s (0:01:46) | REPLAY_RED |
| WB12 | 33 passed, 1 skipped in 48.97s | 0 | 33 passed, 1 skipped in 43.91s | REPLAY_GREEN_WITH_LIMITS |

## Packages C–E
- 24/24 development replay obligations reconciled (verdicts preserved).
- 34/34 NO_NAMED_TEST obligations accounted; mapping artifact ORIGINAL_MAPPING_UNAVAILABLE.
- Wave 2 guard batch executed per T-IGX-090-C command.

## Qualification boundary
Test execution recorded. Independent certification not claimed. G2/R/FCS/M3 not set.
