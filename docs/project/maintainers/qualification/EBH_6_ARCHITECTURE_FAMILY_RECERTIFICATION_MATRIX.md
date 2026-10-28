# EBH-6 — Architecture-Family Recertification Matrix

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_family_catalog.py`](../../../tests/qualification/ebh_6/_family_catalog.py) (`EBH6_FAMILY_CATALOG`) |
| **Independent disposition** | **PENDING (L9)** for most families — not global FRZ PASS |

Per-family columns live in catalog dataclass: Authority, Scope paths, Existing evidence SHA, Freshness, Missing proof, Gate/test IDs, Remediation batch, Acceptance rule, Independent disposition.

## Applicable families (@ HEAD)

| family_id | ecap_theme | frz_family | independent_disposition | L9 proof ref / blocker |
| --- | --- | --- | --- | --- |
| BND | Hard layer boundaries | FRZ-BND-* | REPLAY PASS | `test_hardening_3_layer_boundary_gate.py` tier0 import gate |
| OWN | Exactly-one ownership | FRZ-OWN-* | PENDING | **EBH6-BLK-L9-OWN-01** — catalog gate only |
| CTR | Canonical contracts | FRZ-CTR-* | PENDING | `test_ebh6_no_prohibited_unresolved_in_scope` |
| TYP | Strong typing | FRZ-TYP-* | PENDING | `test_ebh6_tracked_debt_findings_ratified` (Q2-D2) |
| PLG | Pluginability | FRZ-PLG-* | REPLAY PASS | `test_ebh5_exit_inventory_complete` |
| RPL | Replaceability | FRZ-RPL-* | REPLAY PASS | `test_ebh5_rpl_g_config_x_active_blockers_remain_empty` |
| EXE | Execution | FRZ-EXE-* | PENDING | **EBH6-BLK-L9-EXE-01** |
| GOV | Governance | FRZ-GOV-* | PENDING | **EBH6-BLK-L9-GOV-01** |
| STA | State | FRZ-STA-* | PENDING | `test_ebh6_exit_sqlite_annex_accountability` |
| REC | Recovery | FRZ-REC-* | PENDING | **EBH6-BLK-L9-REC-01** |
| CMP | Compatibility | FRZ-CMP-* | N/A — WITH EVIDENCE | COMPAT-X historical CLOSED — no consumer change |
| CFG | Configuration | FRZ-CFG-* | REPLAY PASS | `test_ebh6_config_x_all_concerns_mapped_at_head` |
| SEC | Security | FRZ-SEC-* | PENDING | CTRL-X spot gate @ historical SHA — **EBH6-BLK-L9-SEC-01** |
| REL | Reliability | FRZ-REL-* | PENDING | **EBH6-BLK-L9-REL-01** |
| CTL | Control planes | FRZ-CTL-* | REPLAY PASS | `test_r2_mw_05_fail_open_rejected_before_attach` |
| OBS | Observability | FRZ-OBS-* | PENDING | `test_ebh5_exit_qualified_rows_have_behavioral_refs` (partial) |
| TRC | Traceability | FRZ-TRC-* | PENDING | `test_ebh6_exit_sanctioned_composition_roots_non_empty` |
| PRD | Production qual (arch) | FRZ-PRD-* | N/A — WITH EVIDENCE | Live slice OUT OF SCOPE |
| REG | Regression infra | FRZ-REG-* | PENDING | `test_ebh6_exit_mandatory_docs_present` — not QUAL-X |
| HRN | Harness | FRZ-HRN-* | PENDING | **EBH6-BLK-L9-HRN-01** |
| TEN | Tenant | FRZ-TEN-* | PENDING | **EBH6-BLK-L9-TEN-01** |
| DOC | Documentation | FRZ-DOC-* | PENDING | Doc presence gate — L9 content review |
| VBN | Vendor neutrality rollup | (VBN) | PENDING | Register mechanical gates — not full A–J audit |

**Row count:** 23 applicable families (closed-world catalog).

## Guard / gate replay

Selected pytest bundles per family — see `gate_test_ids` in catalog. Layer boundary gate: [`test_hardening_3_layer_boundary_gate.py`](../../../tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py). EBH-5 RPL-G + inventory integrity replay @ HEAD.

## Explicit non-claims

- Matrix disposition **≠** global [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) PASS rewrite.
- **51** EBH-5 matrix rows **NOT YET QUALIFIED** remain deferred (EBH-7).
- Catalog integrity tests **≠** independent architecture-family PASS.

Pre-cert: [`EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md`](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md).
