# T-IGX-060-A - CURSOR-FACTORY audit follow-ups: Claude surface pin, CLI allowlist, lint, IDE acceptance checklist

## Scope

Close the items left open by the final readiness audit of the Cursor software factory (`T-IGX-059-A`):

1. `tests/unit/governance/test_claude_surface_unchanged.py`: pin the git object hashes of the Claude autonomy surface
   (`scripts/hooks/igx_stop_guard.py`, `governance/agent-control-plane`, `governance/TRUSTED_POLICY_PIN.yaml`,
   `agent-env/skills/igx-run-plan`, `agent-env/skills/igx-autonomy`, `agent-env/rules/intergrax-core.md`,
   `agent-env/agents`, `.claude/settings.json`) and require a clean working tree for them. The pins equal the
   hashes at the factory base `c98d9ac0b` (verified before recording), so they prove byte-identity. Changing one of
   these files on purpose means updating its pin in the same PR (procedure in the test header), which makes any Claude
   autonomy change an explicit, reviewed decision.
2. `.cursor/cli.json`: allow `git checkout`, `git switch` and `git restore` (workers and the integrator use them in
   their own worktrees; headless mode refused them). `git stash` stays unlisted and is denied explicitly: the stash
   stack is shared by every worktree. Extend the `cli.json` test.
3. `scripts/cursor/igx_integrator.py`: wrap two lines over 120 characters (ruff E501) and apply `ruff format` (the push gate requires it for changed
   files). No behaviour change.
4. `docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md`: drop the two stale "known gap" statements about the shell
   guard (closed by `T-IGX-058-A`), document the `cli.json` allowlist, and add the Operator acceptance checklist for
   the in-IDE run. The in-IDE run itself cannot be driven by an agent and stays `CURSOR_IDE_E2E_PENDING_OPERATOR`.

Claude Code files are not edited.

## Plan digest

Plan digest: n/a (no separate plan file; the scope above is the plan)

## Approved

Approved: operator @ 2026-11-22T17:05:00+01:00

Approval source: the Operator wrote "scalone - zrób pozostałe otwarte punkty" (merged - do the remaining open points)
in the Claude Code session on 2026-11-22, after the closeout report that listed exactly these open points. The
timestamp is the recorded entry time. This is not an independently signed IIAC verdict.

## Write set

```text
tests/unit/governance/test_claude_surface_unchanged.py
tests/unit/cursor/test_igx_factory_supervisor.py
.cursor/cli.json
scripts/cursor/igx_integrator.py              (wrapping and formatting only)
docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md
governance/tickets/T-IGX-059-A.md             (lifecycle, checkpoint)
governance/tickets/T-IGX-060-A.md
```

## Test contract

```text
uv run pytest tests/unit/cursor tests/unit/governance/test_claude_surface_unchanged.py tests/unit/governance/test_cursor_native_orchestration.py tests/unit/governance/test_subagent_guard.py tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_stop_guard.py tests/unit/governance/test_hook_commands.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_cursor_stop_guard.py tests/unit/governance/test_cursor_integration_guard.py -q
uv run python scripts/governance/sync_agent_env.py --check
uv run ruff check scripts/cursor tests/unit/cursor tests/unit/governance/test_claude_surface_unchanged.py
```

## Out of scope

`.claude/**`, `CLAUDE.md`, `agent-env/**`, `scripts/hooks/**`, `.cursor/hooks.json`, `governance/agent-control-plane/**`,
`governance/TRUSTED_POLICY_PIN.yaml`, product code. Not re-run: the headless E2E runs A and B predate `T-IGX-058-A`;
the guard change only adds denials and its tests pin the factory commands as allowed.

## Checkpoint

2026-11-22, implementer session:

- Pins recorded from `git rev-parse c98d9ac0b:<path>` and re-checked equal at the branch base `68479a683`.
- `.cursor/cli.json`: allow `git checkout`, `git switch`, `git restore`; deny `git stash`; test extended.
- `igx_integrator.py`: two lines wrapped (E501), no behaviour change.
- Docs: stale guard gaps removed, allowlist documented, in-IDE acceptance checklist added.
- Not re-run: headless E2E (runs A and B). In-IDE E2E stays `CURSOR_IDE_E2E_PENDING_OPERATOR`.

## Lifecycle state

quality_gate
