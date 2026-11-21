# EQO-W3-WB02 — Tenant isolation (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| EQOA-A4-006 | Runtime event tenant propagation / UAEP relocation |
| EQOA-A4-027 | Tier-3 non-strict `tenant_id` silent default |
| EQOA-A7-009 | GAP-TEN-UNK-05 register vs audit matrix |
| EQOA-A7-028 | EBH-7 SCOPED vs open production-delta items |
| EQOA-A7-031 | EBH-7 mandatory tenant E2E skips |
| EQOA-A8-006 | Domain hubs without tenant rules |

## EXE @ TRAIN_PIN

Prereg SHA256: `3047a03f34fa136a81c56fdeb938ccfb93ae311037ade2eeaf254d333f7df1d0` (`DES_PREREG.json`).

**Result:** 66 passed, 1 skipped (Qdrant optional extra). Cross-tenant unit gates for tool invoker, SEC-ENT middleware, TENANT-X P0 fast matrix.

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A4-006 | `ACCEPTED_SCOPED_LIMITATION` — P0 fast + cross-tenant unit gates pass; dedicated UAEP propagation audit not re-bound to current module graph (hypothesis open for session E) |
| A4-027 | `NOT_VERIFIED` — silent-substitution R1 scan not in this pytest slice; no serving-router write set on pin (OD-EQO-08) |
| A7-009 | `ACCEPTED_SCOPED_LIMITATION` — register/label drift; requires doc replay not product patch |
| A7-028 | `ACCEPTED_SCOPED_LIMITATION` — EBH-7 scope residual documented; full E2E replay deferred |
| A7-031 | `BLOCKED` — four tenant E2E skips need routed-proof replay or Operator EBH-7 disposition |
| A8-006 | `ACCEPTED_SCOPED_LIMITATION` — hub doc gap; not closable by pytest alone |

Evidence log: `pytest_wb02_exe.log`. AUD session E pending for all rows.