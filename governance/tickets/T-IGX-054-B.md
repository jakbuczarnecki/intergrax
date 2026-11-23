# T-IGX-054-B — CURSOR-OPT-PARITY P2: parity validators, hooks, docs

## Scope

Plan section 5 (P2): rewrite `scripts/ci/check_cursor_token_setup.py`, stop guard FIX-01, bash guard PowerShell, `.cursor/hooks.json` (`subagentStart` only) with new `scripts/hooks/igx_subagent_guard.py`, `POL-I1-O1` registry pointer, guides and `igx-run-plan` paragraph, matching tests.

Campaign plan: CURSOR-OPT-PARITY (operator-approved via Build; plan file kept outside the repo, digest below). Branch `cursor/opt-parity` from `development`. No PASS, G6 or qualification claim.

## Plan digest

Plan digest: 7ddadcc3dd25b8d2bc21d506c943d238b24e69fea81c8bf854bd34eed13d752b

Canonical plan: `c:\Users\XPS\.cursor\plans\cursor-opt-parity_f1ae8a09.plan.md` (sections 4-6). Computed with `scripts/governance/plan_digest.py`.

## Approved

_(awaiting operator `Approved:` line; the implementer does not write it)_

## Test contract

- `uv run python scripts/governance/sync_agent_env.py --check`
- `uv run python scripts/ci/check_cursor_token_setup.py`
- `uv run pytest tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_read_guard.py tests/unit/governance/test_bash_guard.py tests/unit/governance/test_stop_guard.py tests/unit/governance/test_subagent_guard.py -q`

## Out of scope

`intergrax/**`, `agents/**`, `applications/**`, model routing changes, new subsystems, `.claude/settings.json`, IIAC and G6 gates, git push / PR / merge.

## Lifecycle state

plan_pending
