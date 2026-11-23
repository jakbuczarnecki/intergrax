# T-IGX-064-A — Cursor Direct Execution: remove mandatory software factory orchestration

## Scope

Permanent Cursor-only simplification: drop mandatory parent/worker factory protocol, default integrator/worktree requirements, and factory Stop-hook enforcement. Cursor sessions run as a single direct implementation agent on `development` with local commits in approved ticket scope.

- SSOT rule `igx-cursor-direct-execution` (`loading: cursor-always`) replaces always-on `igx-factory.mdc`.
- `sync_agent_env.py`: new `cursor-always` loading mode; Cursor `intergrax-core` adapter substitutions for direct commit policy (shared `agent-env/rules/intergrax-core.md` unchanged for Claude).
- Remove factory `stop` hook from `.cursor/hooks.json`; keep `subagentStart` and `beforeShellExecution` safety guards.
- Optional voluntary factory scripts/skills remain; not auto-started.
- `.cursor/cli.json`: allow `git push` (guard enforces origin/development only).
- `scripts/hooks/cursor_development_push.py` + guard: allow fast-forward `git push origin development` after preflight.

Claude Code surface, `role_registry.yaml` defaults, enterprise gates, and branch protection unchanged.

## Plan digest

Plan digest: n/a (operator instruction 2026-11-23)

## Approved

Approved: operator @ 2026-11-23T06:45:00+01:00

Authorization: operator message "INTERGRAX — CURSOR ORCHESTRATION REMOVAL" authorizing Cursor Direct Execution Mode on branch `development`.

## Write set

```text
agent-env/rules/igx-cursor-direct-execution.md
agent-env/README.md
scripts/governance/sync_agent_env.py
.cursor/hooks.json
.cursor/cli.json
.cursor/skills/igx-native-run-plan/SKILL.md
.cursor/rules/igx-factory.mdc                    (delete)
tests/unit/governance/test_cursor_native_orchestration.py
tests/unit/governance/test_agent_env_sync.py
tests/unit/governance/test_subagent_guard.py
tests/unit/cursor/test_igx_factory_supervisor.py
governance/tickets/T-IGX-064-A.md
```

Generated adapters (via sync): `.cursor/rules/igx-cursor-direct-execution.mdc`, `.cursor/rules/intergrax-core.mdc`, `.cursor/skills/igx-autonomy/SKILL.md`, `.cursor/skills/igx-run-plan/SKILL.md`, `CLAUDE.md` block (unchanged imports).

## Test contract

```text
uv run pytest tests/unit/governance/test_cursor_native_orchestration.py tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_subagent_guard.py tests/unit/governance/test_claude_surface_unchanged.py tests/unit/cursor/test_igx_factory_supervisor.py -q
uv run python scripts/governance/sync_agent_env.py
uv run python scripts/governance/sync_agent_env.py --check
```

## Out of scope

Claude Code hooks/settings, shared `intergrax-core.md` body text, `role_registry.yaml` global defaults, product runtime code, deleting historical factory documentation under `docs/`, remote push policy.

## Isolation (Claude Code)

- BASE_SHA: `5e59b137dc25419b2c5fab63689a1a41cb53202e`
- Evidence: `.tmp/session/cursor-direct-execution/claude-parity/`
- Fix: `sync_agent_env._claude_md_block` excludes `cursor-always` from Claude on-demand imports; `AGENTS.md` restored to BASE.

## Lifecycle state

plan_approved
## Addendum (2026-11-23)

Cursor Direct Git Push: remove cli deny on push; guarded push to origin/development.
