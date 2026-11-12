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

Approved: operator-delegate @ 2026-11-12T13:53:13+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

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

quality_gate

## Lease holder

Worktree `_wt/intergrax/sfx-p0`, branch `sfx-p0-registration` (base `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`).

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `sfx-p0-registration` (upstream `origin/development`), merged with `origin/development` at `0c241d655` (MCP-SERVER-X closure landed meanwhile; no conflict) |
| Implementers | Orchestrator session plus two implementer subagents (roadmap amendment; SSOT/capability pair/dossier). None of them wrote an `Approved:` line or an OD signature |
| Approvals | Tickets T-IGX-037-A..E: delegate session under DELEGATION-SFX-PRE-FREEZE (`250d3aeeb`). OD-SFX-01..04, 15, 16: signing session under DELEGATION-SFX-OD-SIGNING (`4758c4051`) |
| Test contract | `pytest tests/qualification/roadmap_replay_x tests/unit/governance`: 198 passed, 6 failed, 1 skipped before the inventory fix. The 5 `roadmap_replay_x` failures are pre-existing (same 5 tests fail on an unchanged roadmap; PDC_REG_DIAGNOSIS group C). The 6th (`test_pdc_x_inventory`) was caused by the two new capability docs and fixed by regenerating `PDC_CLOSED_WORLD_INVENTORY.json` (`pdc_x_inventory.py --write`, +2 items). After the merge: `roadmap_replay_x` + `test_pdc_x_inventory` 35 passed, the same 5 pre-existing failures. Logs: `.tmp/session/sfx-p0/` |
| Gates | `sync_agent_env.py --check` OK; `igx_gov ticket` OK (push-fast PASS); pre-commit `check_docs_domain_pairs` OK (4 feature pairs) |
| Writes outside the plan's file list | `PDC_CLOSED_WORLD_INVENTORY.json` (mechanical regeneration only) |
| Product paths | none (`intergrax/`, `agents/`, `applications/`, `scripts/` unchanged); IIAC ACT-N11 unaffected |
