# T-IGX-037-B — SCENARIO-FACTORY-X P1: research, competitive analysis and capability inventory (docs only)

## Scope

Publish the cited competitive evidence matrix, the IDE/MCP tooling facts and the re-pinned closed-world capability inventory for SFX (master plan R1, units SFX-P1-01..03, wave W1).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-B.md`.

## Plan digest

Plan digest: a8142bfebf5c48499b2ff11bc5542b7d67aaf8f50cb8bffd64a1e3ba0fbb2a99

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-B.md`](plans/PLAN_T-IGX-037-B.md).

## Approved

<!-- Awaiting re-approval after plan Amendment 1 (deliverable location, operator decision 2026-11-12). A commissioned delegate session writes `Approved: operator-delegate @ <ISO8601>` + `Delegation:` per DELEGATION-SFX-PRE-FREEZE. The implementer and the orchestrator must not write this line. -->

Superseded approval (plan digest before Amendment 1): see commit `250d3aeeb`.

## Test contract

```text
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- ADRs (T-IGX-037-C), UX (T-IGX-037-D), IDE toolkit (T-IGX-037-E).
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

plan_pending

## Lease holder

Worktree `_wt/intergrax/sfx-w1-research`, branch `sfx-w1-research` (stacked on `sfx-p0-registration`).
