# T-IGX-057-A - CURSOR-FACTORY: autonomous Grok/Composer software factory for Cursor

## Scope

Cursor-only software factory: one Cursor parent chat (e.g. Grok) runs an approved multi-step plan to completion
through Composer subagents without human prompts. Local integration into a per-plan integration branch only (no
push, PR or remote merge inside the loop; the Operator publishes). Park-and-continue escalation. Preflight ticket
approval check at start. Resume via a state-file Cursor `stop` hook (IDE) and a headless `cursor-agent` supervisor.

Canonical plan: `agent-env/plans/cursor-software-factory.md`.

Claude Code behaviour and files stay byte-identical: no edit of `.claude/**`, `CLAUDE.md`, `agent-env/rules/**`,
`agent-env/skills/**`, `agent-env/agents/**`, `scripts/hooks/igx_stop_guard.py`,
`governance/agent-control-plane/**`, `governance/TRUSTED_POLICY_PIN.yaml`, `intergrax/**`, `agents/**`,
`applications/**`.

## Approved

Approved: operator @ 2026-11-22T12:00:00+01:00

Approval source: the Operator accepted the plan `agent-env/plans/cursor-software-factory.md` (plan mode exit) in the
Claude Code planning session on 2026-11-22, after choosing local integration, both resume mechanisms,
park-and-continue and preflight approval. The ISO8601 timestamp is the recorded entry time.

## Write set

```text
scripts/cursor/igx_factory.py
scripts/cursor/igx_test_lock.py
scripts/cursor/igx_factory_supervisor.py
scripts/hooks/igx_factory_stop.py
scripts/hooks/igx_subagent_guard.py            (message text only)
scripts/governance/sync_agent_env.py           (Cursor-only branches)
.cursor/hooks.json
.cursor/cli.json
.cursor/rules/igx-factory.mdc
.cursor/rules/cursor-native-orchestration.mdc  (removed or reduced to pointer)
.cursor/rules/od-af-int-a.mdc                  (alwaysApply: false)
.cursor/rules/intergrax-core.mdc               (generated, Cursor variant)
.cursor/skills/igx-native-run-plan/SKILL.md
.cursor/skills/igx-autonomy/SKILL.md           (generated, Cursor variant)
.cursor/agents/igx-integrator.md
tests/unit/cursor/
tests/unit/governance/test_claude_surface_unchanged.py
tests/unit/governance/test_cursor_native_orchestration.py
tests/unit/governance/test_subagent_guard.py
tests/unit/governance/test_agent_env_sync.py
docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md
docs/project/technical/devos/templates/FACTORY_PLAN_TEMPLATE.md
docs/project/technical/devos/CURSOR_OD_AF_INT_A.md          (superseded pointer)
docs/project/technical/devos/README.md                      (hub link)
docs/project/technical/guides/AGENT_AUTONOMY.md             (Cursor column)
agent-env/plans/cursor-software-factory.md
agent-env/plans/cursor-orch-corrective.md                   (superseded pointer)
governance/tickets/T-IGX-057-A.md
```

## Test contract

```text
uv run pytest tests/unit/cursor tests/unit/governance/test_claude_surface_unchanged.py tests/unit/governance/test_cursor_native_orchestration.py tests/unit/governance/test_subagent_guard.py tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_stop_guard.py tests/unit/governance/test_hook_commands.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_cursor_stop_guard.py -q
uv run python scripts/governance/sync_agent_env.py --check
```

## Out of scope

Remote integration changes (OD-AF-INT A runner stays as is), role registry or policy pin changes, Claude Code
configuration, product code, CI workflows.

## Lifecycle state

quality_gate
