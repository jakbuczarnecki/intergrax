# EBH-6 — CORE Hardcoding Findings Register

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `bc25e27b14b7172ef79951765e76d19578d9beb3` |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) |
| **L9 layer annex** | [`tests/qualification/ebh_6/_layer_evidence.py`](../../../tests/qualification/ebh_6/_layer_evidence.py) |
| **Schema authority** | [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2.2 |

## Mechanical exit (@ qualification SHA)

| Criterion | Status |
| --- | --- |
| `unclassified_findings` (inventory-style) | **0** |
| In-scope `PROHIBITED-*` unresolved | **0** |
| Layer A–J L9 attestation (16 layers) | **RESOLVED @ `_layer_evidence.py`** — independent Protocol 03 still required |

## Finding population (closed-world)

| Category | Count | Notes |
| --- | --- | --- |
| Tracked architecture debts | 2 | Q2-D2 (RATIFIED ADR track), BND-COMPAT-R3 (lab-only) |
| VBN inventory replay | 1 | EBH-5 inventory @ HEAD |
| Layer coverage (§3.0.8) | 16 | `EBH6-LAYER-01..16` — **RESOLVED** with closed-world path inventory + A–J matrix |
| SQLite annex @ HEAD | 38 | `EBH6-SQLITE-*` — mapped from EBH-5 annex |
| **Total** | **57** | Full rows in Python SSOT only |

## Layer status summary (L9 implementer attestation)

| finding_id | layer | status | evidence |
| --- | --- | --- | --- |
| EBH6-LAYER-01..16 | §3.0.8 layers | **RESOLVED** | `_layer_evidence.py` — path inventory, SYS-INV refs, L3 narrative, gate refs |

Spot-check samples **do not** prove global absence of violations — closed-world inventory + mechanical gates only.

## Representative tracked entries

| finding_id | class | classification | status |
| --- | --- | --- | --- |
| EBH6-DEBT-Q2-D2 | F | REQUIRES ARCHITECTURE DECISION | RATIFIED |
| EBH6-DEBT-BND-COMPAT-R3 | H | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RATIFIED |
| EBH6-VBN-INVENTORY-REPLAY | I | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RESOLVED |
| EBH6-LAYER-07 (example) | G | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RESOLVED |

## Reuse (by reference)

- EBH-5 [`EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md`](EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md) — not duplicated.
- EBH-5 [`EBH_5_SQLITE_ANNEX.md`](EBH_5_SQLITE_ANNEX.md) — accountability replay via `_sqlite_touchpoint_registry.py`.

## DUP-X input note

Findings classes **D/E** shadow duplicate mechanisms may overlap **DUP-X** scope — unresolved duplicates must not be silently closed here.

Pre-cert reconciliation: [`EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md`](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md).
