# T-IGX-037-E — SCENARIO-FACTORY-X P4: proposal-only IDE toolkit v0 (agent-env declarative files only)

## Scope

Publish the SFX rule, skills and agent persona briefs in `agent-env/` with their regenerated `.claude/` and `.cursor/` adapters (generator unchanged), plus a walkthrough record (master plan R1, units SFX-P4-A01..A02, wave W2).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-E.md`.

## Plan digest

Plan digest: e733052cf4da700aae879e3edb0d610da983f2a08368a74693debf824cb45c01

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-E.md`](plans/PLAN_T-IGX-037-E.md).

## Approved

Approved: operator-delegate @ 2026-11-12T13:53:13+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

## Test contract

```text
uv run python scripts/governance/sync_agent_env.py --check
uv run pytest tests/unit/governance/test_agent_env_sync.py -q
uv run python scripts/ci/check_cursor_token_setup.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- `scripts/governance/sync_agent_env.py` and `role_registry.yaml` changes (SFX-P4-B01/B02, W3).
- Toolkit v1 (SFX-P10-08); product code, FRZ changes, merges and CLOSED claims.

## Lifecycle state

plan_approved

## Lease holder

Worktree `_wt/intergrax/sfx-w2-ide`, branch `sfx-w2-ide` (stacked on `sfx-p0-registration`).
