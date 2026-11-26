# PLAN T-IGX-077-B — closed-world unknown dependency and CW-12 baseline

An in-universe import with no classified target fails the closed-world gate. The gate does not special-case a test-only prefix. CW-12 reads the frozen qualification baseline, not the live exclusion register.

Write set:

- `testing_support/change_safety/discovery/closed_world.py`
- `tests/qualification/evol_x/data/exclusion_register_cw12_baseline.v1.json`
- `tests/unit/architecture/test_closed_world_tier_gate.py`

Remove `_UNCLASSIFIED_PRODUCT_PREFIXES`. Keep the longest-prefix tier table. `baseline_accepted_exclusion_ids` reads the frozen baseline file. Do not edit the live exclusion register to hide a regression.

Out of scope: product trees, seam ledger, roadmap, FRZ checklist.

Test: `uv run pytest tests/unit/architecture/test_closed_world_tier_gate.py -q --tb=line`
