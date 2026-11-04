# ARCH-FREEZE — VBN freeze SHA dossier (§2.1–2.7 rollup)

**Status:** PLANNING SHELL — **NOT** independently accepted · **NOT** FRZ-RPL promotion  
**FREEZE_CERTIFICATION_SHA:** `TBD` (set at **T-B7-LOCK-FREEZE-SHA**)  
**CLOSURE_DOC_HEAD:** `950097ea4bcfd5bba828f663bcd993b8bd1e1d63`  
**Authority:** [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](./VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2 · WBS **T-B7-VBN-HEAD-DOSSIER** / **T-B7-VBN-INDEPENDENT-ACCEPT**

---

## §2.1 — Vendor/backend dependency inventory

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_vendor_backend_inventory.py`](../../../tests/qualification/ebh_5/_vendor_backend_inventory.py) |
| Mechanical tests | `uv run pytest tests/qualification/ebh_5/ -q` (**L2-VBN-EBH5**) |
| ENT drift gate | `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_inventory_drift.py` |
| Acceptance @ FREEZE_SHA | `unclassified=0`; closed-world concern count reconciled with CONFIG-X historical 54 without equivalence assumption |

## §2.2 — Core hardcoding findings register

| Field | Value |
| --- | --- |
| Evidence modules | EBH-6 qualification evidence under `tests/qualification/` (see VBN doc §2.2) |
| ENT gates | `test_ent_a_vbn_core_hardcoding.py`, shadow/silent/leakage sibling tests |
| Acceptance @ FREEZE_SHA | `prohibited_unresolved=0` in frozen scope |

## §2.3 — Cross-layer replaceability matrix

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_replaceability_matrix.py`](../../../tests/qualification/ebh_5/_replaceability_matrix.py) |
| ENT gate | `test_ent_a_vbn_replaceability_matrix.py` |
| Acceptance @ FREEZE_SHA | RPL-H rows without false `QUALIFIED`; matrix reconciled with inventory |

## §2.4 — Fixed substrate registry

| Field | Value |
| --- | --- |
| SSOT module | [`tests/qualification/ebh_5/_fixed_substrate_registry.py`](../../../tests/qualification/ebh_5/_fixed_substrate_registry.py) |
| Acceptance @ FREEZE_SHA | Fixed substrates **ratified only**; no silent bypass |

## §2.5 — Reachability / production-path proofs

| Field | Value |
| --- | --- |
| EBH-5 proofs | reachability modules under [`tests/qualification/ebh_5/`](../../../tests/qualification/ebh_5/) |
| ENT gate | `test_ent_a_vbn_reachability.py` |
| Acceptance @ FREEZE_SHA | No shadow bypass of sanctioned composition boundaries |

## §2.6 — Behavioral qualification (non–monkeypatch-only)

| Field | Value |
| --- | --- |
| EBH-5 behavioral suites | matrix spot checks per VBN §2.6 |
| Acceptance @ FREEZE_SHA | Behavioral proofs cited for extensibility claims |

## §2.7 — Current-HEAD rollup @ FREEZE_CERTIFICATION_SHA

| Field | Value |
| --- | --- |
| This dossier | Tracked SSOT for ARCH-FREEZE VBN closure @ exact **FREEZE_CERTIFICATION_SHA** |
| ENT VBN domain | `uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py -q` with `ENT_AUDIT_X_SEMANTIC=1` (**L2-VBN-ENT**) — **12/12** required @ FREEZE_SHA |
| Engineering baseline | **T-B2-VBN** COMPLETE @ `f7ce9dae` (engineering ≠ independent acceptance) |
| Acceptance | Independent auditor sign-off; **FIND-VBN-PARTIAL** → **DISPOSITION** only after **T-B7-VBN-INDEPENDENT-ACCEPT** |

---

## Independent acceptance (mandatory)

| Role | Signature | Date | SHA |
| --- | --- | --- | --- |
| independent_auditor | **OPEN** | — | FREEZE_CERTIFICATION_SHA TBD |

**Hard gate:** **T-B7-PROMOTE-RPL** and ECAP §15 RPL-H rows require this sign block @ **FREEZE_CERTIFICATION_SHA** (not **T-B2-VBN** alone).
