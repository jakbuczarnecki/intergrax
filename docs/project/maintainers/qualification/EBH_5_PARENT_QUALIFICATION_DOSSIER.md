# EBH-5 — Parent Qualification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-5 — Replaceability & Vendor/Backend Certification |
| **Baseline HEAD (implementation wave)** | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |
| **AUDIT_BASELINE_SHA** | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |
| **Proof execution SHA** | Recorded at pytest run time via `proof_execution_sha()` (must match git HEAD when suite executed) |
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
| Proof integrity gates | [`test_ebh_5_proof_integrity_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py) |
| Protocol 03 corrective | [EBH_5_PROTOCOL03_CORRECTIVE_RECONCILIATION.md](EBH_5_PROTOCOL03_CORRECTIVE_RECONCILIATION.md) |

## Verification command (L2 + L3)

```powershell
uv run pytest tests/qualification/ebh_5/ -q
```

**Expected @ corrective batch:** 46 passed, 1 skipped (optional RAG graph module). Re-run after any change; do not attribute results to a new SHA without executing the suite at that HEAD.

## FRZ impact

Contributes evidence toward **FRZ-PLG-***, **FRZ-RPL-***, **FRZ-CFG-04/06/08** (current-HEAD). **Does not** promote global FRZ PASS.

## Explicit non-claims

- EBH-6 `CORE_HARDCODING_FINDINGS_REGISTER` not produced here.
- Parent CLOSED requires Protocol 03 + operator publication (see L9 handoff).
