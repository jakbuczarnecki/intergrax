# EBH-6 — Architecture-Family Recertification Matrix

| Field | Value |
| --- | --- |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_family_catalog.py`](../../../tests/qualification/ebh_6/_family_catalog.py) (`EBH6_FAMILY_CATALOG`) |
| **L9 evidence SSOT** | [`tests/qualification/ebh_6/_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py) |
| **Independent disposition** | **23 × PENDING (L9)** — implementer must not self-accept |
| **Implementer proposals** | Overlay from `_family_l9_evidence.py` @ P03 remediation |

## Applicable families (@ HEAD)

| family_id | invariant (summary) | implementer_proposed | evidence nodes (HEAD) | blocker |
| --- | --- | --- | --- | --- |
| BND | Tier import directions | REPLAY PASS (proposed) | `test_hardening_3_layer_boundary_gate` tier0 | — |
| OWN | Exactly-one semantic owner | REPLAY PASS (proposed) | `test_ee_final_arch_owner_matrix_documented` | **EBH6-BLK-L9-OWN-01** |
| CTR | No PROHIBITED unresolved | REPLAY PASS (proposed) | `test_ebh6_prohibited_unresolved_zero` | — |
| TYP | Q2-D2 ratified debt | PASS scoped (proposed) | `test_ebh6_tracked_debt_findings_ratified` | — |
| PLG | EBH-5 inventory replay | REPLAY PASS (proposed) | `test_ebh5_exit_inventory_complete` | — |
| RPL | RPL-G @ HEAD | REPLAY PASS (proposed) | `test_ebh5_rpl_g_config_x_active_blockers_remain_empty` | — |
| EXE | Single execution authority | REPLAY PASS (proposed) | `test_ee_a1_p0_inventory_reports_zero_production_bypass` | **EBH6-BLK-L9-EXE-01** |
| GOV | Governance ≠ execution | REPLAY PASS (proposed) | NPSC42 H1 freeze gates | **EBH6-BLK-L9-GOV-01** |
| STA | SQLite annex classified | PASS scoped (proposed) | `test_ebh6_exit_sqlite_annex_accountability` | — |
| REC | Recovery plane ownership | REPLAY PASS (proposed) | NPSC5e + W3-C4 recovery gates | **EBH6-BLK-L9-REC-01** |
| CMP | COMPAT-X unchanged | N/A — WITH EVIDENCE (proposed) | historical COMPAT-X | — |
| CFG | CONFIG-X 54/54 | REPLAY PASS (proposed) | `test_ebh6_config_x_all_concerns_mapped_at_head` | — |
| SEC | Fail-closed security composition | REPLAY PASS (proposed) | CTRL-X R1 canonical composition | **EBH6-BLK-L9-SEC-01** |
| REL | Capacity ≠ execution owner | REPLAY PASS (proposed) | `test_ee_b1_2_capacity_contract_no_identity_mint` | **EBH6-BLK-L9-REL-01** |
| CTL | Control-plane middleware | REPLAY PASS (proposed) | CTRL-X R2 fail-open rejected | — |
| OBS | Observability non-authority | REPLAY PASS (proposed) | W5-B2 bounded sink wiring | — |
| TRC | Sanctioned composition roots | PASS (proposed) | `test_ebh6_exit_sanctioned_composition_roots_non_empty` | — |
| PRD | Architecture census only | N/A — WITH EVIDENCE (proposed) | PROD-Q Path B scope note | — |
| REG | EBH-6 mechanical gates | PENDING→REPLAY PASS (proposed) | `test_ebh6_exit_mandatory_docs_present` | — |
| HRN | Harness non-authority | REPLAY PASS (proposed) | W5-B2 published event sink | **EBH6-BLK-L9-HRN-01** |
| TEN | Tenant adversarial matrix | REPLAY PASS scoped (proposed) | `test_tenant_scope_adversarial_matrix` | **EBH6-BLK-L9-TEN-01** |
| DOC | Dossier presence | PASS (proposed) | mandatory docs gate | — |
| VBN | Register mechanical exit | REPLAY PASS (proposed) | unclassified=0 + prohibited=0 | — |

**Row count:** 23 applicable families (closed-world catalog).

## Explicit non-claims

- Implementer proposals **≠** independent acceptance.
- Matrix disposition **≠** global FRZ PASS rewrite.
- **51** EBH-5 matrix rows **NOT YET QUALIFIED** remain deferred (EBH-7).

Pre-cert: [`EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md`](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md).  
P03 remediation: [`EBH_6_PROTOCOL_03_REJECTION_REMEDIATION.md`](EBH_6_PROTOCOL_03_REJECTION_REMEDIATION.md).
