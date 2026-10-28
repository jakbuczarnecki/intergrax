# EBH-6 — CORE Hardcoding Findings Register

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) |
| **Schema authority** | [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2.2 |

## Mechanical exit (@ qualification SHA)

| Criterion | Status |
| --- | --- |
| `unclassified_findings` | **0** |
| In-scope `PROHIBITED-*` unresolved | **0** |
| Production-effective `unknown` | **0** |

## Finding population (closed-world)

| Category | Count | Notes |
| --- | --- | --- |
| Tracked architecture debts | 2 | Q2-D2 (RATIFIED ADR track), BND-COMPAT-R3 (lab-only) |
| VBN inventory replay | 1 | EBH-5 inventory @ HEAD |
| Layer coverage (§3.0.8) | 16 | `EBH6-LAYER-01..16` — RESOLVED |
| SQLite annex @ HEAD | 38 | `EBH6-SQLITE-*` — mapped from EBH-5 annex |
| **Total** | **57** | Full rows in Python SSOT only |

## Representative tracked entries

| finding_id | class | classification | status |
| --- | --- | --- | --- |
| EBH6-DEBT-Q2-D2 | F | REQUIRES ARCHITECTURE DECISION | RATIFIED |
| EBH6-DEBT-BND-COMPAT-R3 | H | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RATIFIED |
| EBH6-VBN-INVENTORY-REPLAY | I | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RESOLVED |

## Reuse (by reference)

- EBH-5 [`EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md`](EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md) — not duplicated.
- EBH-5 [`EBH_5_SQLITE_ANNEX.md`](EBH_5_SQLITE_ANNEX.md) — accountability replay via `_sqlite_touchpoint_registry.py`.

## DUP-X input note

Findings classes **D/E** shadow duplicate mechanisms may overlap **DUP-X** scope — unresolved duplicates must not be silently closed here.
