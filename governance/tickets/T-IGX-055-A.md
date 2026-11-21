# T-IGX-055-A — CURSOR-ORCH-CORRECTIVE: Cursor stop adapter, worker lifecycle contract, integrator proposal

## Scope

Cursor-only correction of Grok parent / Composer worker orchestration: new `scripts/hooks/igx_cursor_stop.py`,
Cursor dispatch, root discovery and Cursor plan todos in `scripts/hooks/igx_stop_guard.py`, tests under
`tests/unit/governance/`, Cursor sections of `agent-env/skills/igx-run-plan/SKILL.md` and
`agent-env/skills/igx-autonomy/SKILL.md`, `docs/project/technical/devos/PARALLEL-EXECUTION.md`, campaign plan
and the approval-gated integrator proposal under `agent-env/plans/`.

Campaign plan: `agent-env/plans/cursor-orch-corrective.md`. Branch `fix/cursor-orch-corrective`, stacked on
PR #742 (`ac0c0b753`). No PASS, G6, IIAC or qualification claim.

## Approved

_(awaiting operator `Approved:` line; the implementer does not write it)_

## Test contract

- `uv run python scripts/governance/sync_agent_env.py --check`
- `uv run python scripts/governance/verify_trusted_policy_pin.py`
- `uv run pytest tests/unit/governance/test_stop_guard.py tests/unit/governance/test_cursor_stop_guard.py tests/unit/governance/test_subagent_guard.py tests/unit/governance/test_agent_env_sync.py -q`
- Live Cursor probe recorded under `.tmp/proof/cursor-orch-corrective/` or `CURSOR_LIVE_E2E_NOT_EXECUTED`

## Out of scope

`intergrax/**`, `agents/**`, `applications/**`, `.claude/settings.json`, `CLAUDE.md`,
`governance/agent-control-plane/*.yaml`, `governance/TRUSTED_POLICY_PIN.yaml`, model routing for Claude Code,
`max_parallel` caps, IIAC and G6 gates, git push / PR / merge.

## Lifecycle state

plan_pending
