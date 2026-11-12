# T-IGX-037-D — SCENARIO-FACTORY-X P3: UX design, mock-ups, usability protocol and generalization-set seal (docs only)

## Scope

Publish the SFX UX design with a prototype Artifact, the usability protocol with round-0 baseline, and the sealed generalization set digest (master plan R1, units SFX-P3-01..02, wave W2).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-D.md`.

## Plan digest

Plan digest: a50e0cd805076ba0cab5359d0dfabf894676d24a6fc1309882d8fb831a442452

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-D.md`](plans/PLAN_T-IGX-037-D.md).

## Approved

<!-- Awaiting re-approval after plan Amendment 1 (deliverable location, operator decision 2026-11-12). A commissioned delegate session writes `Approved: operator-delegate @ <ISO8601>` + `Delegation:` per DELEGATION-SFX-PRE-FREEZE. The implementer and the orchestrator must not write this line. -->

Superseded approval (plan digest before Amendment 1): see commit `250d3aeeb`.

## Test contract

```text
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Studio frontend implementation (SFX-P10); revealing the GEN set content.
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

plan_pending

## Lease holder

Worktree `_wt/intergrax/sfx-w2-ux`, branch `sfx-w2-ux` (stacked on `sfx-p0-registration`).
