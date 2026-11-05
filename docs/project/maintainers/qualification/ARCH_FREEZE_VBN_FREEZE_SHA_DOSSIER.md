# ARCH-FREEZE — VBN freeze SHA dossier (§2.1–2.7 rollup)

**Status:** R1 EXPANDED @ baseline HEAD — **NOT** independently accepted · **NOT** FRZ-RPL promotion  
**R1 closure baseline HEAD:** `4a1ab152dc86e97fb017278310c553064a664b23`  
**FREEZE_CERTIFICATION_SHA:** `TBD` (set at **T-B7-LOCK-FREEZE-SHA**; lock **not** authorized in R1)  
**CLOSURE_DOC_HEAD (wave-1 SSOT):** `950097ea4bcfd5bba828f663bcd993b8bd1e1d63`  
**Authority:** [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](./VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2 · WBS **T-B7-VBN-HEAD-DOSSIER** / **T-B7-VBN-INDEPENDENT-ACCEPT** · qual plan **L2-VBN-EBH5** / **L2-VBN-ENT**

**DAG note:** **T-B7-VBN-HEAD-DOSSIER** may proceed @ closure baseline HEAD after **T-B6-OD-ENT-08** (external); certifying **T-B7-VBN-INDEPENDENT-ACCEPT** requires **FREEZE_CERTIFICATION_SHA** after **T-B7-LOCK-FREEZE-SHA**. **T-B7-PFP-ACTIVATE** precedes lock.

---

## §2.1 — Vendor/backend dependency inventory

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_vendor_backend_inventory.py`](../../../tests/qualification/ebh_5/_vendor_backend_inventory.py) |
| Closed-world validation | [`tests/qualification/ebh_5/_inventory_closed_world_validation.py`](../../../tests/qualification/ebh_5/_inventory_closed_world_validation.py) |
| Mechanical tests (**L2-VBN-EBH5**) | `uv run pytest tests/qualification/ebh_5/ -q` |
| ENT drift gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_inventory_drift.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_inventory_drift.py) |
| Status @ `4a1ab152` | Engineering inventory modules present; **no** certifying RUN_ID @ FREEZE_SHA; `unclassified=0` acceptance **pending** independent replay @ FREEZE_SHA |
| Acceptance @ FREEZE_SHA | `unclassified=0`; closed-world concern count reconciled with CONFIG-X historical 54 without equivalence assumption |

## §2.2 — Core hardcoding findings register

| Field | Value |
| --- | --- |
| EBH-5 integrity gates | [`tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py), [`test_ebh_5_proof_integrity_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py), [`test_ebh_5_l2_integrity_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py) |
| ENT gates | [`test_ent_a_vbn_core_hardcoding.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_core_hardcoding.py), [`test_ent_a_vbn_shadow_selection.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_shadow_selection.py), [`test_ent_a_vbn_silent_fallback.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_silent_fallback.py), [`test_ent_a_vbn_contract_leakage.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_contract_leakage.py) |
| Wave-1 engineering evidence | [`ARCH_FREEZE_B2_VBN_CURRENT_HEAD_EVIDENCE.md`](./ARCH_FREEZE_B2_VBN_CURRENT_HEAD_EVIDENCE.md) @ `f7ce9dae` |
| Status @ `4a1ab152` | Dossier paths wired; G6 ENT VBN 12/12 @ scoped HEAD **≠** independent accept @ FREEZE_SHA |
| Acceptance @ FREEZE_SHA | `prohibited_unresolved=0` in frozen scope |

## §2.3 — Cross-layer replaceability matrix

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_replaceability_matrix.py`](../../../tests/qualification/ebh_5/_replaceability_matrix.py) |
| Dual-slug / RPL-EF evidence | [`tests/qualification/ebh_5/_dual_slug_rpl_ef_matrix.py`](../../../tests/qualification/ebh_5/_dual_slug_rpl_ef_matrix.py), [`test_ebh_5_dual_slug_rpl_e_operational_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py) |
| ENT gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_replaceability_matrix.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_replaceability_matrix.py) |
| Status @ `4a1ab152` | Matrix SSOT tracked; certifying matrix replay @ FREEZE_SHA **OPEN** |
| Acceptance @ FREEZE_SHA | RPL-H rows without false `QUALIFIED`; matrix reconciled with inventory |

## §2.4 — Fixed substrate registry

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_fixed_substrate_registry.py`](../../../tests/qualification/ebh_5/_fixed_substrate_registry.py) |
| RPL proofs | [`tests/qualification/ebh_5/_fixed_substrate_rpl_proofs.py`](../../../tests/qualification/ebh_5/_fixed_substrate_rpl_proofs.py) |
| Status @ `4a1ab152` | Registry modules present; ratification sign-off @ FREEZE_SHA **OPEN** |
| Acceptance @ FREEZE_SHA | Fixed substrates **ratified only**; no silent bypass |

## §2.5 — Reachability / production-path proofs

| Field | Value |
| --- | --- |
| EBH-5 reachability gates | [`tests/qualification/ebh_5/test_ebh_5_reachability_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_reachability_gates.py) |
| Runtime platform concerns | [`tests/qualification/ebh_5/_runtime_platform_concerns.py`](../../../tests/qualification/ebh_5/_runtime_platform_concerns.py) |
| ENT gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_reachability.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_reachability.py) |
| Status @ `4a1ab152` | Reachability suites tracked; FREEZE-scope certifying run **OPEN** |
| Acceptance @ FREEZE_SHA | No shadow bypass of sanctioned composition boundaries |

## §2.6 — Behavioral qualification (non–monkeypatch-only)

| Field | Value |
| --- | --- |
| Behavioral registry | [`tests/qualification/ebh_5/_behavioral_proof_registry.py`](../../../tests/qualification/ebh_5/_behavioral_proof_registry.py) |
| EBH-5 behavioral suites | [`test_ebh_5_behavioral_proofs.py`](../../../tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py), [`test_ebh_5_behavioral_package_c.py`](../../../tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py), [`test_ebh_5_behavioral_package_c_v2.py`](../../../tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py) |
| ENT ops coupling | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_obs_ops_coupling.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_obs_ops_coupling.py) |
| Status @ `4a1ab152` | Behavioral modules cited; independent behavioral attestation @ FREEZE_SHA **OPEN** |
| Acceptance @ FREEZE_SHA | Behavioral proofs cited for extensibility claims |

## §2.7 — Current-HEAD rollup @ FREEZE_CERTIFICATION_SHA

| Field | Value |
| --- | --- |
| This dossier | Tracked SSOT for ARCH-FREEZE VBN closure @ exact **FREEZE_CERTIFICATION_SHA** |
| ENT VBN domain (**L2-VBN-ENT**) | `uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py -q` with `ENT_AUDIT_X_SEMANTIC=1` — **12/12** required @ FREEZE_SHA |
| SQLite effective gate | [`tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_sqlite_effective.py`](../../../tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_sqlite_effective.py) |
| Engineering baseline | **T-B2-VBN** COMPLETE @ `f7ce9dae` (engineering ≠ independent acceptance) |
| R1 dossier baseline | Tables §2.1–2.6 populated @ **`4a1ab152`**; **FREEZE_CERTIFICATION_SHA** remains **TBD** |
| Acceptance | Independent auditor sign-off; **FIND-VBN-PARTIAL** → **DISPOSITION** only after **T-B7-VBN-INDEPENDENT-ACCEPT** |

---

## Independent acceptance (mandatory)

| Role | Signature | Date | SHA |
| --- | --- | --- | --- |
| independent_auditor | **OPEN** | — | FREEZE_CERTIFICATION_SHA TBD |

**Hard gate:** **T-B7-PROMOTE-RPL** and ECAP §15 RPL-H rows require this sign block @ **FREEZE_CERTIFICATION_SHA** (not **T-B2-VBN** alone).
