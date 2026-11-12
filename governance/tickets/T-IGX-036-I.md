# T-IGX-036-I — PDC-X P5-REG batch B-REG-T: stale-gate repair (wave B, tests only)

## Scope

Repair 12 stale qualification gates diagnosed in `PDC_REG_DIAGNOSIS.md`:

- `ebh_5`: 4 tests, PDC-D3-R024;
- GR-12: 4 tests, PDC-D4-R020;
- GR-10: 4 tests, PDC-D6-R028.

Also produce a read-only diagnosis record of the other pre-existing red qualification tests (`ebh_7`, `dist_scale_x`, `roadmap_replay_x`). Tests and docs only. Plan: `PLAN_T-IGX-036-I.md`.

## Plan digest

Plan digest: b7d235f47d5f5b3ae02704ebc8ed83be8d9d547fdd3730b6ea8aae201011c494

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-I.md`](plans/PLAN_T-IGX-036-I.md).

## Approved

Approved: operator-delegate @ 2026-11-12T07:46:15+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-B.md@1b49c58cdaa0776e519901aec02e66fb714cc28a

## Test contract

```text
uv run pytest tests/qualification/ebh_5 tests/qualification/governance tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product code.
- The `ollama` regression (B-REG-P).
- Fixes for `ebh_7`, `dist_scale_x` or `roadmap_replay_x`.
- FRZ changes, OD signatures and merges.

## Lifecycle state

plan_approved
