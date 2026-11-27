# T-IGX-090-E — Classify INTELLIGENCE-X docs explicitly in the PDC-X closed-world inventory

## Scope

Fix the `governance-fast` failure caused by `docs/project/capabilities/{architecture,plan}/INTEGRAX_INTELLIGENCE.md` (INTELLIGENCE-X, `T-IGX-079-A`, PROPOSED, not ratified, not certified) being `UNMAPPED` in `scripts/qualification/pdc_x_inventory.py`: add an exact-stem `PLATFORM-FOUNDATION` classification (status `INDEX_ONLY`), correct the module docstring, add classification tests, and regenerate the committed inventory. This fix only.

Linked plan: [`governance/tickets/plans/PLAN_T-IGX-090-E.md`](plans/PLAN_T-IGX-090-E.md)

## Plan digest

Plan digest: 639e1194b59310913a5b8744f112f5801b484702466d22968eddcce6ed66788a

## Approval routing

Approval is not recorded in this ticket. It is a missing authority: the Operator, or the AF-CLOSE signing delegate under OP-A 4 after an independent REVIEW_PASS, must add the approval line. The implementer writes no approval line.

## Test contract

```powershell
uv run python scripts/qualification/pdc_x_inventory.py --check
uv run pytest tests/unit/governance/test_pdc_x_inventory.py -q
uv run ruff check scripts/qualification/pdc_x_inventory.py tests/unit/governance/test_pdc_x_inventory.py
uv run ruff format --check scripts/qualification/pdc_x_inventory.py tests/unit/governance/test_pdc_x_inventory.py
```

## Out of scope

- No product code edits under `intergrax/`, `agents/`, `applications/`.
- No IIAC or certification record edits.
- No representation of INTELLIGENCE-X as ratified or certified.
- Any weakening, `skip` or `xfail` of existing tests.
- Any declaration of `CLOSED`, `IIAC PASS`, `FRZ PASS`, or `G6 PASS`, or any `git push` / merge to `development`.
- C10/C15 environment changes, Docker migration, or vendor qualification changes.

## Lifecycle state

plan_pending
