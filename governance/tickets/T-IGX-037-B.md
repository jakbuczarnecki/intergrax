# T-IGX-037-B — SCENARIO-FACTORY-X P1: research, competitive analysis and capability inventory (docs only)

## Scope

Publish the cited competitive evidence matrix, the IDE/MCP tooling facts and the re-pinned closed-world capability inventory for SFX (master plan R1, units SFX-P1-01..03, wave W1).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-B.md`.

## Plan digest

Plan digest: 4ad9bde5acad82dd3285f71b10e1f6f3763d2fa5ac2c625ce477386b8b646bce

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-B.md`](plans/PLAN_T-IGX-037-B.md).

## Approved

Approved: operator-delegate @ 2026-11-12T13:53:13+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

## Test contract

```text
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- ADRs (T-IGX-037-C), UX (T-IGX-037-D), IDE toolkit (T-IGX-037-E).
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

plan_approved

## Lease holder

Worktree `_wt/intergrax/sfx-w1-research`, branch `sfx-w1-research` (stacked on `sfx-p0-registration`).
