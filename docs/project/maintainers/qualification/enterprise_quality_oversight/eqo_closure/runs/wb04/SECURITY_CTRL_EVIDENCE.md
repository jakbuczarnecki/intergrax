# EQO-W3-WB04 — Security & control planes (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A2-047 | CTRL-X FRZ-CTL-01 (incl. regression protection) is PASS while 15 Security qualification te… |
| A7-013 | PDC-D6-R017 (MCP exposure security) waits on a stage that is already closed; no later sour… |
| A5-015 | ENT-AUDIT FULL certifying run predates 14 security-adjacent product paths merged by T-IGX-… |

## EXE @ TRAIN_PIN

Prereg SHA256: `8b732ccd149bc1b02f63f7ebde4c5ec64add50a2c975f1a732c6c1935781d00a` (`DES_PREREG.json`).

**Result:** 1 failed, 12 passed in 26.70s (exit 1).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A2-047 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A7-013 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A5-015 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |

Evidence log: `pytest_wb04_exe.log`. AUD session E pending for all rows.
