# T-IGX-054-A — CURSOR-OPT-PARITY P1: always-on extract and skill flags

## Scope

Plan section 4 (P1): `igx:active` markers in `agent-env/rules/intergrax-core.md`, `_cursor_rule` projection in `scripts/governance/sync_agent_env.py`, `disable-model-invocation` on `igx-run-plan` and eight `sfx-*` skills, regenerated adapters, `tests/unit/governance/test_agent_env_sync.py`, one paragraph in `agent-env/README.md`.

Campaign plan: CURSOR-OPT-PARITY (operator-approved via Build; plan file kept outside the repo, digest below). Branch `cursor/opt-parity` from `development`. No PASS, G6 or qualification claim.

## Plan digest

Plan digest: 7ddadcc3dd25b8d2bc21d506c943d238b24e69fea81c8bf854bd34eed13d752b

Canonical plan: `c:\Users\XPS\.cursor\plans\cursor-opt-parity_f1ae8a09.plan.md` (sections 4-6). Computed with `scripts/governance/plan_digest.py`.

## Approved

_(awaiting operator `Approved:` line; the implementer does not write it)_

## Test contract

- `uv run python scripts/governance/sync_agent_env.py --check`
- `uv run pytest tests/unit/governance/test_agent_env_sync.py -q`

## Out of scope

`intergrax/**`, `agents/**`, `applications/**`, model routing changes, new subsystems, `.claude/settings.json`, IIAC and G6 gates, git push / PR / merge.

## Lifecycle state

plan_pending
