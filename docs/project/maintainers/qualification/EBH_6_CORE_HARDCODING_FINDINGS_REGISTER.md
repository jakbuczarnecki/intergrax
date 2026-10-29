# EBH-6 — CORE Hardcoding Findings Register

| Field | Value |
| --- | --- |
| **B2 IMPLEMENTATION_SHA** | `8cf61e971a752428ea7eff04e36f1345af123319` |
| **B1 IMPLEMENTATION_SHA (historical)** | `446dfa7263b6571a11bdd2a973e66e57892eacca` |
| **Historical IMPLEMENTATION (P03 remediation)** | `475c20289a1ef0420461a1ba748c792a10ea3e99` — **HISTORICAL** |
| **Closure plan** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) @ `0489054c5f80ee392211fb33a39f2736fa2fdfa0` |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) |
| **L9 layer annex** | [`tests/qualification/ebh_6/_layer_evidence.py`](../../../tests/qualification/ebh_6/_layer_evidence.py) |
| **L3 reachability annex** | [`tests/qualification/ebh_6/_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py) |
| **Schema authority** | [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2.2 |

## Mechanical exit (@ qualification SHA)

| Criterion | Status |
| --- | --- |
| `unclassified_findings` (inventory-style) | **0** |
| In-scope `PROHIBITED-*` unresolved | **0** |
| Layer A–J L9 attestation (16 layers) | **`CANDIDATE-RESOLVED`** @ code SSOT — independent **RESOLVED** only via Protocol 03 |

## Finding population (closed-world)

| Category | Count | Notes |
| --- | --- | --- |
| Tracked architecture debts | 2 | Q2-D2 (RATIFIED ADR track), BND-COMPAT-R3 (lab-only) |
| VBN inventory replay | 1 | EBH-5 inventory @ HEAD |
| Layer coverage (§3.0.8) | 16 | `EBH6-LAYER-01..16` — **`CANDIDATE-RESOLVED`** + L3 mechanism annex |
| SQLite annex @ HEAD | 38 | `EBH6-SQLITE-*` — mapped from EBH-5 annex |
| **Total** | **57** | Full rows in Python SSOT only |

## Layer status summary (L9 implementer attestation)

| finding_id | layer | status | evidence |
| --- | --- | --- | --- |
| EBH6-LAYER-01..16 | §3.0.8 layers | **CANDIDATE-RESOLVED** | `_l3_reachability_evidence.py` — entrypoint, call_chain, A–J per mechanism |

Spot-check samples **do not** prove global absence of violations — closed-world inventory + mechanical gates only.

## Representative tracked entries

| finding_id | class | classification | status |
| --- | --- | --- | --- |
| EBH6-DEBT-Q2-D2 | F | REQUIRES ARCHITECTURE DECISION | RATIFIED |
| EBH6-DEBT-BND-COMPAT-R3 | H | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RATIFIED |
| EBH6-VBN-INVENTORY-REPLAY | I | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | RESOLVED |
| EBH6-LAYER-07 (example) | G | ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM | L9_EVIDENCE_SUBMITTED |

## Reuse (by reference)

- EBH-5 [`EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md`](EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md) — not duplicated.
- EBH-5 [`EBH_5_SQLITE_ANNEX.md`](EBH_5_SQLITE_ANNEX.md) — accountability replay via `_sqlite_touchpoint_registry.py`.

## DUP-X input note

Findings classes **D/E** shadow duplicate mechanisms may overlap **DUP-X** scope — unresolved duplicates must not be silently closed here.

Pre-cert reconciliation: [`EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md`](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md).
