# PLAN T-IGX-036-B — PDC-P1 closed-world inventory

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md) §G "PDC-P1". Roadmap stage `PDC-X` (§3.2). No product code.

## Deliverables

1. **PDC-P1-01 (Pass A, structural index):** add `scripts/qualification/pdc_x_inventory.py`, a deterministic read-only scanner. It emits `docs/project/maintainers/qualification/platform_domain_completeness/PDC_CLOSED_WORLD_INVENTORY.json`, which covers:
   - every top-level package under `intergrax/` and `intergrax/runtime/`, plus `agents/` and `applications/`;
   - every `docs/project/architecture/*.md`, `docs/project/maintainers/plans/*.md` and `docs/project/capabilities/**/*.md`;
   - every `tests/qualification/*` directory.

   Each item is assigned to one of `PDC-D1`..`PDC-D8`, `PLATFORM-FOUNDATION` (owned by an already-certified non-PDC stage) or `UNMAPPED`. It also gets a coverage status (`INSPECTED`, `INDEX_ONLY`, `UNINSPECTED`) from a checked-in classification table. The scan excludes `applications/*/docker/runtime-context/**`. `--check` mode fails if any item is `UNMAPPED`, or if the committed JSON differs from a fresh scan.
2. **PDC-P1-02:** for each domain, link architecture doc ↔ plan ↔ semantic owner (code path) ↔ composition/config owner ↔ qualification suite ↔ last accepted certification SHA. Published in `PDC_CLOSED_WORLD_INVENTORY.md`.
3. **PDC-P1-03:** semantic inspection of `docs/project/architecture/intergrax_runtime_architecture.md`, `INTERGRAX_ARCHITECTURE_PRINCIPLES.md` §14–§15, and `HARNESS_TOP_TIER_GAP_AUDIT.md` §16 ("DO NOT REBUILD"), recorded with their D1–D8 implications. This resolves finding PDC-BL-F05.
4. **PDC-P1-04:** supersession chains for Protocol 03, the operator decision registers, the SHA ladders and FRZ promotion.
5. Unit test `tests/unit/governance/test_pdc_x_inventory.py`. It checks that the scan is deterministic, that the vendored copies are excluded, that the check mode rejects `UNMAPPED`, and that the committed JSON matches.

## Acceptance

- Zero `UNMAPPED` mandatory surfaces.
- Every `UNINSPECTED` item is listed as a tracked unknown and carried into P2.
- The JSON is reproducible (`--check` exit 0).

## Test contract

```text
uv run python scripts/qualification/pdc_x_inventory.py --check
uv run pytest tests/unit/governance/test_pdc_x_inventory.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; requirement dispositions (P2); FRZ edits; MCP-SERVER-X files; IIAC protocol and scripts.
