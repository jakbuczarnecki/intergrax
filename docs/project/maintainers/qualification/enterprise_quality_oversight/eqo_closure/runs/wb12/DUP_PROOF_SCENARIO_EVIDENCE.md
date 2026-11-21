# EQO-W3-WB12 — DUP, proof receipts & scenarios (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A3-019 | Row-level PASS evidence for EXE/GOV/CTL/REC/HRN is tied to certification-time SHAs; substa… |
| A4-039 | Two unrelated ProofReceipt models and two ProofReceiptStore definitions share names across… |
| A4-047 | Scenario-facing contracts have no single semantic owner: four scenario mechanisms with ove… |
| A5-010 | DUP-X duplicate-mechanism freedom not re-examined for 70 product modules added since the a… |
| A8-020 | Accepted Protocol-v2 remediation (LCI trusted tenant scope, Token Optimization receipt ide… |
| A2-031 | NPSC-4.2 allowlist 'FAIL retained' text vs allowlist gate refreshed under EBH-7 W3 without… |

## EXE @ TRAIN_PIN

Prereg SHA256: `cee7f64c024da3331c708a8f6482e8317f79709c9af49bae9a32ae81b3c400ea` (`DES_PREREG.json`).

**Result:** 33 passed, 1 skipped in 65.97s (0:01:05) (exit 0).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A3-019 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A4-039 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A4-047 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A5-010 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A8-020 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |
| A2-031 | `ACCEPTED_SCOPED_LIMITATION — band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone (AUD session E pending)` |

Evidence log: `pytest_wb12_exe.log`. AUD session E pending for all rows.
