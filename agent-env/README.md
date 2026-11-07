# agent-env - shared agent environment (Cursor · Claude Code)

Single source of truth for everything an AI coding agent needs that is **not** tool-specific. The tool folders `.cursor/` and `.claude/` contain **only thin adapters** that point here.

```text
agent-env/
  rules/<id>.md            canonical rules (frontmatter: description, loading)
  skills/<id>/SKILL.md     canonical skills (frontmatter: name, description)
  plans/                   plan-mode output (Claude: plansDirectory; Cursor: move by hand)
AGENTS.md                  tool-neutral project stub (Claude imports it via CLAUDE.md)
CLAUDE.md                  Claude-only rules + generated import block
.cursor/rules/<id>.mdc     GENERATED  Cursor rule  -> @agent-env/rules/<id>.md
.cursor/skills/<id>/       GENERATED  Cursor skill -> agent-env/skills/<id>/SKILL.md
.claude/skills/<id>/       GENERATED  Claude skill -> agent-env/skills/<id>/SKILL.md (+ one per non-always-on rule)
.claude/settings.json      Claude shared settings (plansDirectory, ask/deny permissions)
.cursorignore, .cursorindexingignore   Cursor-only settings; must stay at the repo root
```

## Rules and how each tool loads them

`loading` in the rule frontmatter drives the adapters:

| `loading` | Cursor | Claude Code |
|-----------|--------|-------------|
| `always` | `.cursor/rules/<id>.mdc`, `alwaysApply: true` | `@agent-env/rules/<id>.md` import block in `CLAUDE.md` |
| `explicit` | `.mdc`, `alwaysApply: false` (use `@`) | skill `/<id>`, `disable-model-invocation: true` (operator only) |
| `auto` | `.mdc`, `alwaysApply: false` (agent decides) | skill `/<id>`, model may invoke |

| Rule | `loading` |
|------|-----------|
| `intergrax-token-budget`, `intergrax-test-process-budget`, `intergrax-session-artifacts`, `intergrax-git-branch-safety`, `intergrax-env-vendor-qualification` | always |
| `intergrax-iteration`, `intergrax-hep-step`, `intergrax-ci-hotfix`, `intergrax-micro-implement` | explicit |
| `intergrax-agent-roles`, `intergrax-independent-audit` | auto |

Skills (`igx-devos-*`) are routed to roles by [`governance/agent-control-plane/skill_routing_registry.yaml`](../governance/agent-control-plane/skill_routing_registry.yaml) and exist in both tools under the same id.

## Workflow

- **Add or change a rule/skill:** edit only `agent-env/`, then `uv run python scripts/governance/sync_agent_env.py`. Never edit generated adapters (they carry a `GENERATED` marker; hand-written files in `.cursor/` / `.claude/` are left alone).
- **Check for drift (CI, `igx_gov doctor`):** `uv run python scripts/governance/sync_agent_env.py --check`.
- **New skill:** also add it to `skill_routing_registry.yaml` and a role's `allowed_skills` (`verify_agent_skills.py`).
- Rule text is written in Cursor vocabulary (`@file`, `Task`, `explore`); the mapping to Claude Code is in `CLAUDE.md`.

## Legacy paths

Older evidence documents (`docs/project/maintainers/qualification/**`, `governance/tickets/plans/**`) still cite the former locations; they are historical and are not rewritten:

| Former | Now |
|--------|-----|
| `.cursor/rules/<id>.mdc` | `agent-env/rules/<id>.md` |
| `.cursor/skills/<id>/SKILL.md` | `agent-env/skills/<id>/SKILL.md` |
| `.cursor/plans/` | `agent-env/plans/` |
