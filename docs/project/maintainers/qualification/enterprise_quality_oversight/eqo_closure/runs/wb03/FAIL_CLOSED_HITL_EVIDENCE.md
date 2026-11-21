# EQO-W3-WB03 — Fail-closed & HITL (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A4-026 | Inbound interaction request verification defaults to a no-op verifier; unset or invalid mo… |
| A4-031 | UAEP path skips canonical context assembly when context_engine or llm_adapter is unset; CE… |
| A4-037 | Nexus-internal HITL decision persistence is a silent no-op in the production Nexus constru… |
| A3-027 | FRZ-PRD-07 PASS candidate cannot be confirmed: strict-path evidence partial and strict fix… |

## EXE @ TRAIN_PIN

Prereg SHA256: `dcab81279fde5468ab5890e5423861ac7cdc95a843dc92cc8793d8c599318fcb` (`DES_PREREG.json`).

**Result:** 39 passed in 92.18s (0:01:32) (exit 0).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A4-026 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A4-031 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A4-037 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A3-027 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |

Evidence log: `pytest_wb03_exe.log`. AUD session E pending for all rows.
