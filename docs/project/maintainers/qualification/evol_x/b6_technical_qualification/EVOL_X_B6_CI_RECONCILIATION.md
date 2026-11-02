# EVOL-X B6 — CI qualification reconciliation

**Gate:** G-01 · **Run ID:** `41337077334`

## Published record

Authoritative row: [EVOL_X_CI_QUALIFICATION_RECORD.md](../EVOL_X_CI_QUALIFICATION_RECORD.md)

| Field | Value |
|-------|-------|
| TES | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| Conclusion | success (2026-11-01) |
| Tests | 290 passed |
| VL3 | 290 node ids |
| JUnit SHA256 | `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` |
| Workflow | Regression gate / `evol_x_p3_l3` |
| Authorization | OD-EVOL-10; scoped semantics OD-EVOL-13 |

## Independent gh verification (@ E3 consolidation)

```
gh run view 41337077334 → conclusion: success
Workflow log: TEST_EXECUTION_SHA=3c4a641b…; git rev-parse HEAD equality assertion
```

## Relationship to Mode B

CI qualifies **exact-TES** execution on Linux runners. Local Mode B replay qualifies **outcome integrity** against FINAL archive @ auditor environment. Neither substitutes for the other; both **PASS** for AUTOMATION_FIRST_SCOPED E3 technical track.

## Limitation

Local copy `.tmp/session/evol-x-w6-04-requal-3c4a641/junit.xml` not present — see F-002.

## Disposition

**G-01 technical and formal CI qualification PASS** (per published record + gh metadata).
