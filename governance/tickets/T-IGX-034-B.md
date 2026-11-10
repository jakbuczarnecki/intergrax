# T-IGX-034-B — PRE-ENT Wave 1: HIST-F-BND-03B product manifest registry (Tier-0 → Tier-3 dynamic edge)

## Scope

Remove the Tier-0 dynamic import of Tier-3 manifest modules from `intergrax/applications/_shared/product_manifest_registry.py` (`importlib.import_module` on `legal_application.manifest`, `research_application.manifest`, `dispute_sim_application.manifest`, `local_workspace_application.manifest`) without weakening BND-03 gates.

Deliverables:

- Tier-0 in-memory manifest registry (`register_product_manifest`, `iter_product_manifests`, `iter_strict_product_manifests`).
- Tier-3 `applications/shipped_product_manifest_catalog.py` with static manifest imports and registration at import time.
- Regression gate `tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py`.
- Bootstrap imports for scripts/tests (and doctor CLI handlers) so conformance gates keep the same product coverage.

Worktree slug: `bnd-03b`. Base: `development` @ `4a932d0cf`.

No PASS, FRZ promotion, IIAC closure, or global BND-03 PASS claim.

## Plan digest

Plan digest: 2823002ee0f5fbbdb8f8fa4363cf076d4c23bedfbf99e0c9b3767939c5e3a6a6

## Approved

Approved: operator-delegate pre-ent-wave1 @ 2026-11-10T09:15:00+01:00 | Delegation: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE0_PRESENCE_RECONCILIATION.md@4a932d0cf5487b72e433805e372e3855fcd3d987

## Test contract

One blocking pytest before handoff:

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py -q
```

## Out of scope

Weakening or allow-listing `test_bnd03_tier0_upward_edges_gate.py`; relocating doctor commands to Tier-3; FRZ checklist promotion; ENT-AUDIT-X; other Wave 1 tickets (034-C..I); push/merge/PR unless operator requests.

## Lifecycle state

quality_gate

## Checkpoint

Implementation complete (2026-11-10). `igx_gov ticket`: PASS push-fast (5.2s). Test: `test_bnd03b_product_manifest_registry_gate.py` — 2 passed.

| Item | Status |
|------|--------|
| HIST-F-BND-03B dynamic `importlib` in registry | **removed** — Tier-3 catalog registers manifests |
| BND-03 CLI gate | **unchanged** |
| FRZ-BND-03 global PASS | **not claimed** (residual: doctor CLI still bootstraps catalog via dynamic import) |

**Local `development` integration (2026-11-10):** fast-forward merge `t-igx-034-b` → `development` @ `01b3e1961f01e1ad578b84f3a20fbde7e16755f6`; no conflicts; not pushed. Post-merge pytest `test_bnd03b_product_manifest_registry_gate.py` — 2 passed, exit 0. Evidence: [`WAVE1_BND03B_STATUS.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_BND03B_STATUS.md). Lifecycle remains `quality_gate` pending operator-directed audit routing (no `ready_for_audit` self-promotion).

**operator-delegate pre-ent-wave2 (2026-11-10):** pytest pass @ HEAD `f2682d651f436c19f7000f940647be8fd134afa4` — still **NOT** `ready_for_audit`, no stage closure. Evidence: [`WAVE2_SERIAL_QUAL_LOG.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md). Gate: `tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py` — 2 passed, exit 0.
