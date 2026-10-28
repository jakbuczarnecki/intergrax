# EBH-5 — Parent Qualification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-5 — Replaceability & Vendor/Backend Certification |
| **Baseline HEAD (implementation)** | `5c9b66db632976ada039d406474530d7c6cb5f86` |
| **Status** | **READY FOR INDEPENDENT STAGE CERTIFICATION** — **not** `CLOSED — INDEPENDENTLY ACCEPTED` ([readiness](EBH_5_STAGE_CERTIFICATION_READINESS.md)) |
| **DIST-SCALE-X** | `CLOSED — OPERATOR SCOPED ACCEPTANCE` @ `df485a962688c4fdc4c4e95fea4005d614578c2d` (unchanged) |

## Artifact index

| Artifact | Location |
| --- | --- |
| Inventory | [EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md](EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md) |
| Matrix | [EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md](EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md) |
| Reachability | [EBH_5_VENDOR_SELECTION_REACHABILITY.md](EBH_5_VENDOR_SELECTION_REACHABILITY.md) |
| Behavioral proofs | [`test_ebh_5_behavioral_proofs.py`](../../../tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py) |
| Fixed substrates | [EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md](EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md) |
| SQLite annex | [EBH_5_SQLITE_ANNEX.md](EBH_5_SQLITE_ANNEX.md) |
| RPL-G | [`test_ebh_5_rpl_g_regression_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py) |
| CONFIG-X reconciliation | [`_config_x_reconciliation.py`](../../../tests/qualification/ebh_5/_config_x_reconciliation.py) |

## Verification command (L2 + L3)

```powershell
uv run pytest tests/qualification/ebh_5/ -q
```

**Expected @ baseline:** 24 passed, 1 skipped (optional RAG graph module).

## FRZ impact

Contributes evidence toward **FRZ-PLG-***, **FRZ-RPL-***, **FRZ-CFG-04/06/08** (current-HEAD). **Does not** promote global FRZ PASS.

## Explicit non-claims

- EBH-6 `CORE_HARDCODING_FINDINGS_REGISTER` not produced here.
- Parent CLOSED requires Protocol 03 + operator publication (see L9 handoff).
