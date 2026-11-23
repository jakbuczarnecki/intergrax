# EQO-W3-WB10 — Trace, state & roadmap replay (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A2-019 | TRACE-X-P6 certified restart/resume surface changed after the accepted evidence SHA with n… |
| A2-030 | DUP-X, ROADMAP-REPLAY-X (and EBH-7 GHA run) acceptance rests on auditor log inspection of … |
| A5-012 | STATE-X and TRACE-X closed-world gates were not executed after the 2026-10-30 replay altho… |

## EXE @ TRAIN_PIN

**Wave-10 rerun (2026-11-21):** 6 failed, 66 passed, 1 skipped in 98.73s (0:01:38) (exit 1). TRACE-X closed-world and ROADMAP-REPLAY mechanical exit gates still red on TRAIN_PIN.
Prereg SHA256: `e70afb0eeecc8a182df8e993afb77c4e84140c5889ad6350a761770ea5150d64` (`DES_PREREG.json`).
**Result:** 6 failed, 66 passed, 1 skipped in 80.56s (0:01:20) (exit 1).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A2-019 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A2-030 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A5-012 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |

Evidence log: `pytest_wb10_exe.log`. AUD session E pending for all rows.
