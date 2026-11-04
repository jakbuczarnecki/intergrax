# ENT + EVOL — Formal closure publication evidence

**Disposition:** preparatory publication record — **not** formal acceptance  
**START_HEAD:** `0e538269a7188075003b9ff2c7cce2545d8f1ccb`  
**Branch:** `development` · **remote:** `https://github.com/jakbuczarnecki/intergrax.git`

## SHA references (active ENT qualification)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` |
| EVIDENCE_PUBLICATION_SHA | `300859537f12a1de7bf3d75fd685bc3a5c727183` |
| INDEPENDENT_TECHNICAL_REAUDIT_SHA | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` |
| FROZEN ENT AUDIT HEAD (nightly FULL) | `618e234ee4dc41aa3673de5e1e81271f6039ca6c` |

## EVOL pinned evidence (handoff)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |

## PUBLICATION_SHA

`b14a409c9894c77257487364a2d42d63ec22e295` (`FORMAL_B6_PACKAGE_PREPARATION_SHA`)

## Deliverables published (paths)

| ID | Document |
| --- | --- |
| A | [ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md](./ent_audit_x/ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md) |
| A′ | [ENT_AUDIT_X_INDEPENDENT_TECHNICAL_REAUDIT_REPORT_DRAFT.md](./ent_audit_x/ENT_AUDIT_X_INDEPENDENT_TECHNICAL_REAUDIT_REPORT_DRAFT.md) |
| A″ | [ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md](./ent_audit_x/ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md) |
| B | [EVOL_X_FORMAL_CLOSURE_RECONCILIATION_PACKAGE_DRAFT.md](./evol_x/EVOL_X_FORMAL_CLOSURE_RECONCILIATION_PACKAGE_DRAFT.md) |
| C | [ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md](./ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md) |
| D | [ENT_EVOL_OPERATOR_SIGNATURE_PACKET_DRAFT.md](./ENT_EVOL_OPERATOR_SIGNATURE_PACKET_DRAFT.md) |

## Drift corrections (ENT-REAUD-W8-007)

- [ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md](./ent_audit_x/ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md) — certifying qualification @ TES `89c76eb7…`
- [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ent_audit_x/ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) — active header pins
- [ENT_AUDIT_X_R8_EXTERNAL_GATE_PACKET.md](./ent_audit_x/ENT_AUDIT_X_R8_EXTERNAL_GATE_PACKET.md) — W8 active vs historical pins

## GitHub CI evidence (verified via `gh run view`)

| Run ID | head_sha | Result |
| --- | --- | --- |
| 41800553851 | `618e234…` | success (ENT nightly FULL) |
| 41804655934 | `0e538269…` | success (regression gate) |
| 41798742568 | `461bfc84…` | success (FAST) |

## Validation (L1/L2)

| Check | Result @ prep |
| --- | --- |
| `mechanical_readiness_check.py` | All listed gates OK @ START_HEAD |
| B5 strict verifier | Via mechanical readiness `evidence_integrity` |
| Ancestry TES → HEAD | PASS |
| Immutable R7 evidence | Unmodified under `qualification_at_87203c1…` |

## Internal certification (operator — OD-IC-01)

Operator-authorized **internal** qualification acceptance (2026-11-04) is recorded separately and **does not** replace formal B6 attestation or FRZ promotion:

- [`EVOL_ENT_OD_IC_01_OPERATOR_INTERNAL_ACCEPTANCE_2026_11_04.md`](./EVOL_ENT_OD_IC_01_OPERATOR_INTERNAL_ACCEPTANCE_2026_11_04.md)
- [`EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md`](./EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md)

## Remaining blockers

- ENT formal B6 attestation (independent auditor)
- EVOL formal B6 + FRZ-EVOL + closure SHA
- OD-ENT-00/01/02/08 and OD-EVOL-11 as applicable
- GitHub branch protection (admin)
- **BLK-ROADMAP-ENT-DEP** until OD-ENT-00

## Next responsible actor

1. **Authorized independent auditor(s)** — EVOL B6 then ENT B6 (or parallel review per OD-ENT-00 option)  
2. **Operator** — OD-ENT-00, OD-EVOL-11, post-B6 OD chain  
3. **GitHub administrator** — OD-ENT-02 enforcement  

**Terminal disposition target:** `ENT+EVOL FORMAL PACKAGES PUBLISHED — READY FOR AUTHORIZED AUDITOR/OPERATOR DECISIONS` (after commit+push confirms `PUBLICATION_SHA`).
