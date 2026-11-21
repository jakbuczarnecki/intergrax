# EQO-W3-WB11 — Compat & PROD-Q (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A6-007 | CHANGED-SINCE is fail-closed only for the audit tooling prefixes; a new product package pa… |
| A7-037 | 164 skip/xfail markers in qualification and architecture test trees are not in the EBH-7 s… |
| A1-008 | Roadmap-declared FRZ closing links for older stages are not reflected in the FRZ checklist… |
| A1-012 | COMPAT-X-R5 closure rests on a single gate; full COMPAT suite declared NOT RUN |
| A1-015 | PROD-Q-P0 replay result (438/436/0/2) rests on an untracked .tmp log; P2/P3 closed with no… |
| A2-023 | P0/P2/P3 B6 closures, P0-R1 x8 row and Path B amendment lack independent exact-SHA audit; … |
| A2-024 | P0 closed-world inventory gate not replayed after 202 product-path commits since its B6 re… |
| A2-051 | COMPAT-X-R2-P0-R2 was accepted with 47 failed / 1633 passed in its targeted run recorded a… |
| A7-003 | FRZ-DEBT-04/05 reconciliation proves 'no new' rows, not a zero inventory of existing PARTI… |
| A7-007 | Zero-PROHIBITED and zero-duplicate counts in hardcoding/DUP-X registers are pinned to pre-… |

## EXE @ TRAIN_PIN

**Wave-10 rerun (2026-11-21):** 1 failed, 68 passed in 188.10s (0:03:08) (exit 1). Primary failure: `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`.
Prereg SHA256: `088a4be905ca5dfbd81a39aa1b86c853d37d346cc4698e37b9342318bcffb9ef` (`DES_PREREG.json`).
**Result:** 1 failed, 68 passed in 150.74s (0:02:30) (exit 1).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A6-007 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A7-037 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A1-008 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A1-012 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A1-015 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A2-023 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A2-024 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A2-051 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A7-003 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A7-007 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |

Evidence log: `pytest_wb11_exe.log`. AUD session E pending for all rows.
