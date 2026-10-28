# EBH-5 — Parent Qualification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-5 — Replaceability & Vendor/Backend Certification |
| **Baseline HEAD (implementation wave)** | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |
| **AUDIT_BASELINE_SHA** | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |
| **Corrective publication** | `ffe0aae8d2b64a558a18226cc1435f0058ff561b` |
| **F01 OTLP composition** | `3e9460465d46a64f82022d37130891d940a810e2` |
| **CERTIFICATION_HANDOFF_SHA** | [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) |
| **TEST_EXECUTION_SHA** | Recorded when pytest suite executed (must match git HEAD at run) |
| **Status** | **DOSSIER COMPLETE — INDEPENDENT AUDIT PENDING** — **not** `CLOSED — INDEPENDENTLY ACCEPTED` ([readiness](EBH_5_STAGE_CERTIFICATION_READINESS.md)) |
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
| Protocol 03 final correction | [EBH_5_PROTOCOL03_FINAL_CORRECTION.md](EBH_5_PROTOCOL03_FINAL_CORRECTION.md) |
| Certification completion | [EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) |
| L9 handoff | [EBH_5_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_5_L9_INDEPENDENT_AUDIT_HANDOFF.md) |

## Verification command (L2 + L3)

```powershell
uv run pytest tests/qualification/ebh_5/ -q
```

**Expected @ CERTIFICATION_HANDOFF_SHA:** run pytest and record exact counts in [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) — do not reuse stale numbers without re-run.

## FRZ impact

Contributes evidence toward **FRZ-PLG-***, **FRZ-RPL-***, **FRZ-CFG-04/06/08** (current-HEAD). **Does not** promote global FRZ PASS.

## Explicit non-claims

- EBH-6 `CORE_HARDCODING_FINDINGS_REGISTER` not produced here.
- Parent CLOSED requires Protocol 03 + operator publication (see L9 handoff).
