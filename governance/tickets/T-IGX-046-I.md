# T-IGX-046-I — PT-G01 closed-world tier gate and seam ledger correctness (EC-V3 S-W03b)

## Scope

ARCH-FREEZE train corrective for closed-world tier classification (F02), CW-12 exclusion baseline (F05), seam import extraction (F03), and missing-seam detection in `run_seam_ledger_check` (F04). Target branch: `arch-freeze/rw-train`. Basis: `be0ce70d39b0d811e96177410ac5f9d27c770c5c`.

Sibling tickets keep their write sets: `T-IGX-046-H` (AST bypass gate only), `T-IGX-046-F` on `development` (documentation-only).

## Plan digest

Plan digest: pending

Canonical plan: none (operator-authorized EC-V3 S-W03b package scope).

## Approved

Approved: operator @ 2026-11-24T18:05:00+01:00

## Lifecycle state

plan_approved

## Write set

```text
governance/architecture/seam_ledger.yaml
governance/tickets/T-IGX-046-I.md
scripts/maintenance/check_closed_world_tier_gate.py
scripts/maintenance/check_seam_ledger.py
scripts/maintenance/seam_ledger_lib.py
testing_support/change_safety/discovery/closed_world.py
testing_support/change_safety/discovery/closed_world_gates.py
tests/qualification/evol_x/data/exclusion_register_cw12_baseline.v1.json
tests/unit/architecture/test_closed_world_tier_gate.py
tests/unit/architecture/test_seam_ledger_gate.py
```

## Test contract

One process at a time under `D:/Projekty/_wt/_locks/pytest.lock` (do not pass `-p no:cacheprovider`):

- `uv run pytest tests/unit/architecture/test_closed_world_tier_gate.py tests/unit/architecture/test_seam_ledger_gate.py -q --tb=line`

## Out of scope

`scripts/maintenance/check_agent_registry_bypass.py`, seam/closed-world governance documentary batches, product paths, FRZ promotion, independent verification claims.