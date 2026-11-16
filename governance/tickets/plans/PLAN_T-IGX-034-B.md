## PLAN `T-IGX-034-B` — PRE-ENT Wave 1: BND-03B product manifest registry edge

Campaign: PRE-ENT predecessor certification closure (Wave 1 parallel batch `bnd-03b`).

### 1. Defect

`intergrax/applications/_shared/product_manifest_registry.py` hard-codes four Tier-3 manifest module names and resolves them with `importlib.import_module` (HIST-F-BND-03B). Tier-0 must not name or dynamically import Tier-3 modules in this registry.

### 2. Fix (alternative b — push-tier catalog)

| Layer | Change |
|-------|--------|
| Tier-0 registry | In-memory `register_product_manifest` + `iter_*` over registered manifests only; no `importlib`, no Tier-3 string literals |
| Tier-3 catalog | New `applications/shipped_product_manifest_catalog.py` statically imports the four shipped manifests and registers them at import time |
| Callers | Scripts under `scripts/` and `tests/unit/applications/conftest.py` import the catalog after `sys.path` setup; Tier-0 doctor CLI loads catalog via `importlib.import_module("shipped_product_manifest_catalog")` inside command handlers (CLI scope unchanged from 033-A) |
| Gate | New `tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py` — AST scan of registry module only (no allow-list) |

BND-03 CLI gate (`test_bnd03_tier0_upward_edges_gate.py`) is **not** weakened.

### 3. Files

**Product:** `intergrax/applications/_shared/product_manifest_registry.py`, `applications/shipped_product_manifest_catalog.py`, `intergrax/cli/doctor_health_app.py`, `intergrax/cli/doctor_diff_app.py`, maintenance/gate scripts that iterate manifests (catalog bootstrap import only).

**Tests:** `tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py`, `tests/unit/applications/conftest.py`.

**Governance:** this plan, `T-IGX-034-B.md`.

### 4. Test contract

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py -q
```

### 5. Out of scope

FRZ-BND-03 PASS promotion, widening BND-03 gate to all of `intergrax/`, CLI doctor relocation to Tier-3, ARCH-FREEZE edits, PASS/closure, push/merge/PR unless operator requests.

### 6. Rollback

Revert product + test + script bootstrap commits; restore dynamic registry.
