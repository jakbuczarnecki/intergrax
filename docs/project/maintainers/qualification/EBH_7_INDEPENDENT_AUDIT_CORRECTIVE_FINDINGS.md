# EBH-7 — Independent audit corrective findings

**Audited SHA:** `59d0fe5d7faf271926f8477de102e7f26b3e1caf`  
**Verdict:** INDEPENDENT AUDIT FAIL — qualification integrity correction applied (behavioral gates remain OPEN where proofs missing)

## Corrected false-positive gates

| Gate ID | Was | Now |
| --- | --- | --- |
| W6-ID-REGISTRY-ONLY | 26 IDs = E2E PASS | Structural registry + executed TENANT-X proofs; 11 scenarios OPEN |
| W6-TENANT-X-FINAL-READY | Single readiness test | Removed as E2E substitute |
| W7-HASATTR-DUP-X | hasattr only | Execute proof callables; 17 XDM scenarios OPEN |
| W5-RPL-ROW-COUNT | 56 rows = RPL PASS | 51 NOT YET remain QUALIFICATION_PENDING |
| W5-VBN-DOC-PRESENCE | Docs exist = VBN PASS | MATERIALIZED only |
| TRACE-466-COUNT | Count only | Extended rows + behavioral integrity gate |
| READINESS-TABLE-PASS | Wave PASS labels | Materialization vs qualification columns |
| GAP-TEN-UNK-05-CLOSED | “Closed” implied E2E | Structural N/A evidence only |

SSOT: `tests/qualification/ebh_7/_gate_findings_register.py`

## Certification status semantics

`INVENTORIED` → `PLANNED` → `MATERIALIZED` → `QUALIFICATION_PENDING` → `QUALIFIED` → `READY_FOR_INDEPENDENT_AUDIT` → `INDEPENDENTLY_ACCEPTED`

Registry/doc presence = **MATERIALIZED**, not **QUALIFIED**.

## Remaining operator / independent obligations

- **OD-EBH7-08:** OPEN  
- **Protocol 03 B6:** OPEN  
- **51× RPL-H NOT YET** @ HEAD  
- **12× tenant E2E** + **16× cross-domain** scenarios without current-HEAD proof nodes  
