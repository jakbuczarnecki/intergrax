# EVOL-X — Internal acceptance contract (OD-IC-01 extension)

**Campaign:** `EVOL-X-INTERNAL-FINAL-CLOSURE` · **Task:** IC-01  
**Authorization:** `OPERATOR-DIRECTED INTERNAL CLOSURE CAMPAIGN` (2026-11-05)  
**Prior operator record:** [OD-IC-01](../EVOL_ENT_OD_IC_01_OPERATOR_INTERNAL_ACCEPTANCE_2026_11_04.md) (2026-11-04) — preserved, not superseded  
**Agent role:** `internal-stage-closure-auditor` (separate Cursor session) — **not** independent Protocol 03 signatory

## Contract dimensions (required statuses)

| Field | Value |
| --- | --- |
| `INTERNAL_AUDIT_STATUS` | **AUTOMATED INTERNAL AUDIT ACCEPTED (AUTOMATION_FIRST_SCOPED)** |
| `INTERNAL_B6_TECHNICAL_STATUS` | **INTERNAL_B6_TECHNICAL_ACCEPTED** |
| `INTERNAL_FRZ_STATUS` | **INTERNAL_FRZ_EVOL_01..10 = ACCEPTED_SCOPED** |
| `INTERNAL_PARENT_CLOSURE_STATUS` | **EVOL-X — INTERNALLY CLOSED — AUTOMATION_FIRST_SCOPED** |
| `FORMAL_ECAP_B6_STATUS` | **PENDING_AUTHORITY** (`FORMAL_ECAP_INDEPENDENT_B6 = PENDING_AUTHORITY`) |
| `FORMAL_FRZ_STATUS` | **FRZ-EVOL FORMAL — OPEN** (checklist rows unchanged) |
| `FORMAL_PARENT_CLOSURE_STATUS` | **NOT** `CLOSED — INDEPENDENTLY ACCEPTED`; `FORMAL_CLOSURE_PUBLICATION_SHA` **unset** |

## Evidence pins (immutable — no requalification)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI | `41337077334` |
| R2 reconciliation | `167583ad1d33d9eb6e82589661e6d18aee15fa92` |
| Local B6 unattended | `42363732d9f450bfe4329d76a96c646d2ab730b2` |

## Scope rules

1. Preserve enterprise invariants ([SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md)).
2. Preserve exact-SHA traceability via [EVOL_X_SHA_LADDER.md](./EVOL_X_SHA_LADDER.md).
3. **AUTOMATION_FIRST_SCOPED:** **EVOL-M-09A = MEASURED**; **M-09H = NOT_MEASURED**.
4. **FRZ-EVOL-07:** **Outcome A** (qualification-layer deploy/rollback assessor only — **not** live production rollback enforcement).
5. Internal acceptance **must not** populate `B6_ACCEPTANCE_PUBLICATION_SHA` or `FORMAL_CLOSURE_PUBLICATION_SHA`.
6. Word **INDEPENDENT** reserved for formal ECAP Protocol 03 attestation only.
7. Formal OD records (OD-EVOL-11..19 formal slots) retain historical status; internal companions are distinct.

## Publication

| Role | Record |
| --- | --- |
| Internal B6 | [EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md](./EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md) |
| Internal FRZ | [EVOL_X_INTERNAL_FRZ_EVOL_ACCEPTANCE_MATRIX.md](./EVOL_X_INTERNAL_FRZ_EVOL_ACCEPTANCE_MATRIX.md) |
| Internal parent | [EVOL_X_INTERNAL_PARENT_CLOSURE_RECORD.md](./EVOL_X_INTERNAL_PARENT_CLOSURE_RECORD.md) |
| `INTERNAL_CLOSURE_PUBLICATION_SHA` | Ladder row (pin commit after publication) |
