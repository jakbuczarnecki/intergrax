# T-IGX-075-A — ARCH-FREEZE P-R0-A: node ledger factual recount (development pins)

## Scope

Documentation-only **P-R0-A / P0 / V4** recount of [`ARCH_FREEZE_NODE_LEDGER.md`](../../docs/project/maintainers/qualification/arch_freeze_program/ARCH_FREEZE_NODE_LEDGER.md) (149 WP-12 / MEP rows). **Current pins:** `origin/development` `5935fa4f0cd3596852b3a6500f9d4e1d613a53df` (canonical integration head; not FCS, not G2) and `origin/arch-freeze/rw-train` `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a`. V4 pin `7657745c38500653c57980b0b477f611a86cccdc` and P0 pin `11787a7e196cbfea32a982bfa260f02a34fe05fc` are historical. V5 adds the MEP-AF2 Appendix V5 execution amendment and refreshes the read pin. **Does not** promote `SCOPE`, `G2`, `FCS`, `M3`, or `ARCH_FREEZE_CLOSED`.

## Write set

```text
docs/project/maintainers/qualification/arch_freeze_program/ARCH_FREEZE_NODE_LEDGER.md
docs/project/maintainers/qualification/arch_freeze_program/continuation/CHECKPOINT.md
docs/project/maintainers/qualification/ARCH_FREEZE_SHA_LADDER.md
docs/project/maintainers/qualification/arch_freeze_program/MEP-AF2_v2.1.md
governance/tickets/T-IGX-075-A.md
```

## Approved

Approved: operator-delegate @ 2026-11-24T17:40:00+01:00
Delegation: c:\Users\XPS\.cursor\plans\arch-freeze_completion_b13ee733.plan.md (Pakiet A autonomy; campaign ARCH-FREEZE)

## Test contract

None (docs-only). If this ticket file changes in the same commit batch: `uv run python -m scripts.governance.igx_gov ticket`.

## Out of scope

FRZ checklist, roadmap, product paths, train merge to `development`, status upgrades beyond existing closure records, pytest.

## Lifecycle state

in_progress