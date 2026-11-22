# T-IGX-059-A - CURSOR-FACTORY closeout: Cursor core rule vs local integration, ticket lifecycles

## Scope

Close the gaps found by the final readiness audit of the Cursor software factory (`T-IGX-057-A`, `T-IGX-058-A`, both
merged into `development` at `7bd4558b5`):

1. The always-on Cursor adapter `.cursor/rules/intergrax-core.mdc` still says "merge, branch integration ... stay
   human (forbidden action: escalate)" and "Stop only with `ESCALATION`". That contradicts the factory's local
   integration into `factory/<plan-id>/int` and park-and-continue, and can stall a Cursor parent at its first local
   merge. Fix with Cursor-only substitutions in `scripts/governance/sync_agent_env.py` (`_cursor_core_text`, same
   fail-loudly drift check as the existing PLAN COMPLETE substitution), then regenerate. Push, PR, remote merge and
   integration into `development` or operator branches stay human.
2. Set the lifecycle of `T-IGX-057-A` and `T-IGX-058-A` to `closed` (both merged) and record the merge evidence.

Claude Code stays byte-identical: canonical `agent-env/rules/intergrax-core.md` and `CLAUDE.md` are not edited.

## Plan digest

Plan digest: n/a (no separate plan file; the scope above is the plan)

## Approved

Approved: operator @ 2026-11-22T16:10:00+01:00

Approval source: the Operator answered "Yes, fix it here" in the Claude Code session on 2026-11-22 to the proposal to
fix the core-rule conflict and close out both tickets under a new ticket. The timestamp is the recorded entry time.
This is not an independently signed IIAC verdict.

## Write set

```text
scripts/governance/sync_agent_env.py          (Cursor-only substitutions)
.cursor/rules/intergrax-core.mdc              (generated)
tests/unit/governance/test_agent_env_sync.py
governance/tickets/T-IGX-057-A.md             (lifecycle, checkpoint)
governance/tickets/T-IGX-058-A.md             (lifecycle, checkpoint)
governance/tickets/T-IGX-059-A.md
```

## Test contract

```text
uv run pytest tests/unit/cursor tests/unit/governance/test_claude_surface_unchanged.py tests/unit/governance/test_cursor_native_orchestration.py tests/unit/governance/test_subagent_guard.py tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_stop_guard.py tests/unit/governance/test_hook_commands.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_cursor_stop_guard.py tests/unit/governance/test_cursor_integration_guard.py -q
uv run python scripts/governance/sync_agent_env.py --check
```

## Out of scope

`.claude/**`, `CLAUDE.md`, `agent-env/**`, `scripts/hooks/**`, `scripts/cursor/**`, `.cursor/hooks.json`,
`.cursor/cli.json`, `governance/agent-control-plane/**`, `governance/TRUSTED_POLICY_PIN.yaml`, product code. Audit
items left open: tree-hash pin in `test_claude_surface_unchanged.py`, `cli.json` allowlist for
`checkout`/`switch`/`restore`/`stash`, E501 in `scripts/cursor/igx_integrator.py`, in-IDE E2E
(`CURSOR_IDE_E2E_PENDING_OPERATOR`).

## Closeout

2026-11-22: merged into `development` via PR #751 (merge commit `68479a683`). Verification before push:
ticket test contract 318 passed, `sync_agent_env.py --check` OK, `igx_gov ticket` OK. The audit items left open here are
handled by `governance/tickets/T-IGX-060-A.md`.

## Lifecycle state

closed
