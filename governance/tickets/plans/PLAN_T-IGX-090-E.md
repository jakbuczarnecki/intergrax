# Plan: T-IGX-090-E — Classify INTELLIGENCE-X docs explicitly in the PDC-X closed-world inventory

## Objective

Fix the `governance-fast` failure at `development` 7d2cc3d28 (`test_committed_inventory_is_current_and_closed_world`, `test_every_item_has_exactly_one_known_category_and_status`). The INTELLIGENCE-X documents `docs/project/capabilities/architecture/INTEGRAX_INTELLIGENCE.md` and `docs/project/capabilities/plan/INTEGRAX_INTELLIGENCE.md` (ticket `T-IGX-079-A`) are classified `UNMAPPED` by `classify_doc`, and the committed inventory is stale.

1. Add an exact-stem table `DOC_FOUNDATION_EXACT_STEMS = frozenset({"INTEGRAX_INTELLIGENCE"})` to `scripts/qualification/pdc_x_inventory.py`, consulted in `classify_doc`, mapping to `PLATFORM-FOUNDATION` with status `INDEX_ONLY` (not added to `INSPECTED`). No broad `INTEGRAX_` prefix; the closed world is preserved.
2. Correct the module docstring: `PLATFORM-FOUNDATION` is outside PDC semantic scope, owned by a non-PDC stage, domain pair or program (certified or not); `INDEX_ONLY` carries no certification claim. Add a code comment: INTELLIGENCE-X is PROPOSED (not ratified, not certified); classification is index-only.
3. Add tests: exact stem maps to foundation; `INTEGRAX_UNKNOWN_FUTURE_DOC` and `TOTALLY_UNKNOWN_DOC` stay `UNMAPPED`; the Intelligence architecture doc item is `INDEX_ONLY` in the scanned payload.
4. Regenerate the committed inventory with `--write`; the diff changes only the Intelligence items and summary counts.

## Edit Scope (Strict)

- `governance/tickets/T-IGX-090-E.md`
- `governance/tickets/plans/PLAN_T-IGX-090-E.md`
- `scripts/qualification/pdc_x_inventory.py`
- `tests/unit/governance/test_pdc_x_inventory.py`
- `docs/project/maintainers/qualification/platform_domain_completeness/PDC_CLOSED_WORLD_INVENTORY.json`

## Out of Scope

- Any product code edit under `intergrax/`, `agents/`, `applications/`.
- Any IIAC or certification record edit.
- Any representation of INTELLIGENCE-X as ratified or certified (it is PROPOSED).
- Any weakening, `skip` or `xfail` of existing tests; any `git push` / merge to `development`.
- C10/C15 environment changes, Docker migration, or vendor qualification changes.

## Verification Contract (Serial)

1. `uv run python scripts/qualification/pdc_x_inventory.py --check`
2. `uv run pytest tests/unit/governance/test_pdc_x_inventory.py -q`
3. `uv run ruff check scripts/qualification/pdc_x_inventory.py tests/unit/governance/test_pdc_x_inventory.py`
4. `uv run ruff format --check scripts/qualification/pdc_x_inventory.py tests/unit/governance/test_pdc_x_inventory.py`
