# EBH-6 — Architecture-Family Recertification Matrix

| Field | Value |
| --- | --- |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_family_catalog.py`](../../../tests/qualification/ebh_6/_family_catalog.py) (`EBH6_FAMILY_CATALOG`) |
| **Independent disposition** | **23 × PENDING (L9)** — implementer must not self-accept |
| **Implementer proposals** | **`implementer_proposed_disposition`** @ B-L9-3 — auditor field unchanged |

## Applicable families (@ HEAD)

| family_id | implementer_proposed | independent_disposition | L9 proof ref / blocker |
| --- | --- | --- | --- |
| BND | REPLAY PASS (proposed) | PENDING | tier0 import gate |
| OWN | PENDING (proposed) | PENDING | **EBH6-BLK-L9-OWN-01** |
| CTR | PENDING (proposed) | PENDING | prohibited unresolved gate |
| TYP | PENDING (proposed) | PENDING | Q2-D2 ratified |
| PLG | REPLAY PASS (proposed) | PENDING | EBH-5 inventory replay |
| RPL | REPLAY PASS (proposed) | PENDING | RPL-G @ HEAD |
| EXE | PENDING (proposed) | PENDING | **EBH6-BLK-L9-EXE-01** |
| GOV | PENDING (proposed) | PENDING | **EBH6-BLK-L9-GOV-01** |
| STA | PASS scoped (proposed) | PENDING | sqlite annex gate |
| REC | PENDING (proposed) | PENDING | **EBH6-BLK-L9-REC-01** |
| CMP | N/A — WITH EVIDENCE (proposed) | PENDING | COMPAT-X historical |
| CFG | REPLAY PASS (proposed) | PENDING | CONFIG-X 54/54 |
| SEC | PENDING (proposed) | PENDING | **EBH6-BLK-L9-SEC-01** |
| REL | PENDING (proposed) | PENDING | **EBH6-BLK-L9-REL-01** |
| CTL | REPLAY PASS (proposed) | PENDING | fail-open middleware gate |
| OBS | PENDING (proposed) | PENDING | qualified rows partial |
| TRC | PASS (proposed) | PENDING | sanctioned roots gate |
| PRD | N/A — WITH EVIDENCE (proposed) | PENDING | live OUT OF SCOPE |
| REG | PENDING→REPLAY PASS (proposed) | PENDING | mandatory docs gate |
| HRN | PENDING (proposed) | PENDING | **EBH6-BLK-L9-HRN-01** |
| TEN | REPLAY PASS scoped (proposed) | PENDING | **EBH6-BLK-L9-TEN-01** |
| DOC | PASS (proposed) | PENDING | dossier presence |
| VBN | PENDING (proposed) | PENDING | register mechanical + L9 layers |

**Row count:** 23 applicable families (closed-world catalog).

## Explicit non-claims

- Implementer proposals **≠** independent acceptance.
- Matrix disposition **≠** global FRZ PASS rewrite.
- **51** EBH-5 matrix rows **NOT YET QUALIFIED** remain deferred (EBH-7).

Pre-cert: [`EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md`](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md).
