# EQO-W3-WB09 — Skills & serving (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A2-027 | SKILL-X certified surface changed after acceptance; F-04 scope note obsolete and a PROD-Q … |
| A4-040 | Most Tier-3 application entrypoints start serving with direct uvicorn.run instead of the h… |

## EXE @ TRAIN_PIN

Prereg SHA256: `a9361ab1914706309bc7081f553d335b021854a2a2f808233efc1ba7f0461030` (`DES_PREREG.json`).

**Result:** 1 failed, 65 passed in 11.30s (exit 1).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A2-027 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A4-040 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |

Evidence log: `pytest_wb09_exe.log`. AUD session E pending for all rows.
