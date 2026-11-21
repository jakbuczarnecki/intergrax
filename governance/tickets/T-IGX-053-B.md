# T-IGX-053-B — AGENT-OPS-X W1: core rules, settings, agents, hooks

## Scope

Packets W1-P01, W1-P02, W1-P03, W1-P04, W1-P05, W1-P06, W1-P07, W1-P08 of campaign AGENT-OPS-X (wave branch `agent-ops/w1`). Edit scope is the campaign charter `edit_globs` only. No PASS, FRZ promotion or qualification claim.

## Plan digest

Plan digest: 17f6ec53a29eebe6224a7d18d0bd02b09bd95800c2c28494cd001b1448502163

Canonical plan: `governance/campaigns/AGENT-OPS-X/source/plan-v2.1.md`. Computed with `scripts/governance/plan_digest.py`.

## Approved

Approved: operator @ 2026-11-20 (AGENT-OPS-X v2.1 plan approval; bootstrap legacy form)

## Test contract

- `uv run pytest tests/unit/governance/test_agent_env_sync.py -q`
- `uv run python scripts/governance/sync_agent_env.py --check`
- `uv run pytest tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_agent_role_routing.py -q`
- `uv run python -m scripts.governance.agent_ops.probe overhead`
- `uv run pytest tests/unit/governance/test_read_guard.py tests/unit/governance/test_bash_guard.py -q`
- `uv run python -m scripts.governance.agent_ops.boot run W1-P08`

## Out of scope

`intergrax/**`, `agents/**`, `applications/**`, `agent-env/plans/**`, `.github/**`, the IIAC activation scripts and the IIAC certification protocol (charter `forbidden_globs`).

## Lifecycle state

in_progress
