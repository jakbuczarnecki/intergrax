# T-IGX-037-A — SCENARIO-FACTORY-X P0: roadmap registration and SFX documentation baseline (docs only)

## Scope

Register `SCENARIO-FACTORY-X — Enterprise Scenario Factory` in the canonical roadmap as `PLANNED / MANDATORY / POST-ARCH-FREEZE PRIORITY`, placed directly after ARCH-FREEZE. The registration must not change any existing predecessor: SFX is not a predecessor of SCENARIO-READINESS-X or SCENARIO-GATE, and the OD-SEF-02 parallelism is kept.

This batch also publishes:

- the approved master plan (`SFX-MASTER-PLAN.md`),
- the capability doc pair,
- the qualification dossier skeleton,
- the operator decision records OD-SFX-01..04, which are pending signature.

Documentation and governance records only. Plan: `PLAN_T-IGX-037-A.md`.

## Plan digest

Plan digest: d07c7ad43f6f5b2da43575e39314750e100d084ddc4d4cb027d2fe5fc14134ec

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-A.md`](plans/PLAN_T-IGX-037-A.md).

## Approved

<!-- Awaiting a commissioned delegate session (`Approved: operator-delegate @ <ISO8601>` + `Delegation:`) per DELEGATION-SFX-PRE-FREEZE, or the operator. The implementer and the orchestrator must not write this line. -->

## Test contract

```text
uv run pytest tests/qualification/roadmap_replay_x tests/unit/governance -q
uv run python scripts/governance/sync_agent_env.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product code (`intergrax/`, `agents/`, `applications/`), `scripts/` and `agent-env/` changes.
- Competitive research findings (SFX-P1), ADRs (SFX-P2) and the IDE toolkit (SFX-P4).
- Fixes for findings RA-F-01 and RA-F-03..06.
- FRZ changes, OD signatures by the implementer, merges, and any CLOSED claim.

## Lifecycle state

plan_pending

## Lease holder

Worktree `_wt/intergrax/sfx-p0`, branch `sfx-p0-registration` (base `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`).
