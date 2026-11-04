# EVOL-X + ENT-AUDIT-X — OD-IC-01 — Operator internal acceptance

**Date:** 2026-11-04  
**Project/repository:** INTEGRAX — `jakbuczarnecki/intergrax`  
**Branch:** `development`  
**Authority:** Explicit operator disposition in project conversation (internal certification model; no external trust system at this program stage).

## Decision

The operator approves adoption of the **internal certification model** for the current development stage and registers **operator-accepted internal technical qualification** for **EVOL-X** and **ENT-AUDIT-X**, based on existing technical evidence, qualification runs, technical re-audit material, and prior operator decisions — **without** substituting for ECAP-mandated independent Protocol 03 B6 attestation or freeze checklist promotion.

**Disposition:** `OPERATOR APPROVED — INTERNAL CERTIFICATION RECORDED (EVOL-X + ENT-AUDIT-X)`

## EVOL-X — internal certification

**Internal certification status:** `INTERNALLY CERTIFIED — OPERATOR ACCEPTED (SCOPED)`

**Basis (unchanged authoritative pins):**

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI | run `41337077334` (VL3 290/290 @ TES) |
| Prior operator chain | OD-EVOL-17, OD-EVOL-18, OD-EVOL-19 |

**Formal independent certification status:** **PENDING** — unsigned Protocol 03; `B6_ACCEPTANCE_PUBLICATION_SHA` unset; `FORMAL_CLOSURE_PUBLICATION_SHA` unset; **FRZ-EVOL-01..10** remain **OPEN** per checklist.

## ENT-AUDIT-X — internal certification

**Internal certification status:** `INTERNALLY CERTIFIED — OPERATOR ACCEPTED (TECHNICAL SCOPE)`

**Basis (unchanged authoritative pins):**

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` |
| EVIDENCE_PUBLICATION_SHA (B5) | `300859537f12a1de7bf3d75fd685bc3a5c727183` |
| INDEPENDENT_TECHNICAL_REAUDIT_SHA | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` |
| Technical re-audit disposition | **TECHNICAL B6 PASS RECOMMENDED** (recommendation only) |
| Nightly FULL CI | run `41800553851` @ FROZEN ENT AUDIT HEAD `618e234ee4dc41aa3673de5e1e81271f6039ca6c` |
| Regression gate | run `41804655934` @ `0e538269…` |
| W8 findings | [`ENT_AUDIT_X_W8_B6_FINDINGS_DISPOSITION.md`](./ent_audit_x/ENT_AUDIT_X_W8_B6_FINDINGS_DISPOSITION.md) |

**Formal independent certification status:** **PENDING** — authorized independent Protocol 03 ENT-AUDIT-X B6 attestation not recorded; `B6_ACCEPTANCE_PUBLICATION_SHA` unset; `FORMAL_CLOSURE_PUBLICATION_SHA` unset; **FRZ-AUD-01..10** remain **OPEN** per checklist.

## Governance boundaries (preserved)

| Gate | State |
| --- | --- |
| ECAP independent B6 (EVOL / ENT) | **Not satisfied** by this decision |
| `B6_ACCEPTANCE_PUBLICATION_SHA` (EVOL / ENT) | **unset** |
| `FORMAL_CLOSURE_PUBLICATION_SHA` (EVOL / ENT) | **unset** |
| Roadmap formal stage `CLOSED — INDEPENDENTLY ACCEPTED` | **Not claimed** |
| **ARCH-FREEZE** | **Not promoted** |
| Formal B6 packages / dossiers | **Preserved** — not revoked |

No fictional auditor identity, signature, or external trust mechanism is created by this record.

## Operator acceptance evidence

This document is the primary operator acceptance artifact for OD-IC-01. Consolidated cross-stage record: [`EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md`](./EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md).

**START_HEAD (pre-publication):** `becbcb9c4339e081b9cc3094c84d3fcd595e1ffe`  
**INTERNAL_CERTIFICATION_PUBLICATION_SHA:** `5c08e5e6c44d21998c76bd27ad1efab172440f45`
