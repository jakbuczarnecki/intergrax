# T-IGX-037-D — SCENARIO-FACTORY-X P3: UX design, mock-ups, usability protocol and generalization-set seal (docs only)

## Scope

Publish the SFX UX design with a prototype Artifact, the usability protocol with round-0 baseline, and the sealed generalization set digest (master plan R1, units SFX-P3-01..02, wave W2).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-D.md`.

## Plan digest

Plan digest: a50e0cd805076ba0cab5359d0dfabf894676d24a6fc1309882d8fb831a442452

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-D.md`](plans/PLAN_T-IGX-037-D.md).

## Approved

Approved: operator-delegate @ 2026-11-12T15:06:03+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

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

quality_gate

## Lease holder

Worktree `_wt/intergrax/sfx-w2-ux`, branch `sfx-w2-ux` (stacked on `sfx-p0-registration`).

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `sfx-w2-ux` (upstream `origin/development`). It contains `origin/development` @ `0140bc6b5` and the whole W1 branch `sfx-w1-research` (the skills reference the W1 schemas), so this PR stays stacked on PR 486 until that one merges |
| Implementers | Orchestrator session and implementer subagents; the GEN set was written by the independent session `sfx-gen-author`. None of them wrote an `Approved:` line. Approval: delegate re-approval after plan Amendment 1 (`85ab15622`) |
| Test contract | `sync_agent_env.py --check` OK (57 adapters + CLAUDE.md block); `test_agent_env_sync.py` 2 passed; `check_cursor_token_setup.py` OK; `igx_gov ticket` OK; `pdc_x_inventory.py --check` OK (354 items, 0 UNMAPPED). Log: `.tmp/session/sfx-w2/contract.log` |
| Acceptance | 17 screens SCR-01..17 and the journeys cover all 11 lifecycle steps; the clickable prototype Artifact is linked from `SFX_UX_DESIGN.md`; usability protocol has every result field `NOT_RUN` (round 0 not run); GEN set sealed by the independent session (6 problems, 6 domains, digest in `SFX_GEN_SET_SEAL.md`, content kept outside the repository). Open UX questions UX-Q-01..06 are carried into the gap analysis |
| Product paths | none changed: no `scripts/`, `intergrax/`, `agents/`, `applications/` or `governance/agent-control-plane/` file differs from `development`; IIAC ACT-N11 unaffected |
