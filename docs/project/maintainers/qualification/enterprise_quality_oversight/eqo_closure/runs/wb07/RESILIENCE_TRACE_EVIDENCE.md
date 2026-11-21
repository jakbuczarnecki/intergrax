# EQO-W3-WB07 — Resilience & trace (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A2-043 | Certified surfaces changed after closure without requalification record (W4 permit/cancell… |
| A2-049 | P6 restart/resume classifier defaults whole directories to canonical resume, so 'unclassif… |
| A4-034 | Dormant and duplicate retry-budget mechanisms next to the active inline budget |
| A4-045 | Artifact reference vocabularies coexist beside DurableArtifactRef without a stated mapping… |

## EXE @ TRAIN_PIN

Prereg SHA256: `cf4a7b80c122e4fe960ee810731ae2ed13245e185da032b56e07744a3862d605` (`DES_PREREG.json`).

**Result:** 1 failed, 56 passed, 1 skipped in 69.73s (0:01:09) (exit 1).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A2-043 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A2-049 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A4-034 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A4-045 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |

Evidence log: `pytest_wb07_exe.log`. AUD session E pending for all rows.
