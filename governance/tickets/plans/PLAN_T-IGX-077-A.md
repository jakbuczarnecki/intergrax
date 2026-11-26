# PLAN T-IGX-077-A — live seam ledger missing-seam gate

The live seam check must report a cross-boundary import whose source sits outside every scan root, and must deny a parent package when only a child package is allowed.

Write set:

- `governance/architecture/seam_ledger.yaml`
- `scripts/maintenance/seam_ledger_lib.py`
- `scripts/maintenance/check_seam_ledger.py`
- `tests/unit/architecture/test_seam_ledger_gate.py`

Both live rules `agents_no_direct_integrations` and `lower_layers_no_applications` set `report_missing_seam_outside_scan_roots: true`. `run_seam_ledger_check` calls `detect_unregistered_cross_boundary_import`. A source outside every `scan_roots` that imports a denied prefix, or that prefix plus a dotted suffix, is `MISSING_SEAM`.

Out of scope: T-IGX-046-H, product trees, roadmap, FRZ checklist.

Test: `uv run pytest tests/unit/architecture/test_seam_ledger_gate.py -q --tb=line`
