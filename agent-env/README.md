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
.claude/settings.json      Claude shared settings (plansDirectory, attribution off, ask/deny permissions)
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
| `intergrax-core` | always |
| `intergrax-iteration`, `intergrax-hep-step`, `intergrax-ci-hotfix`, `intergrax-micro-implement` | explicit |
| `intergrax-agent-roles`, `intergrax-independent-audit`, `intergrax-env-vendor-qualification`, `intergrax-scenario-factory` | auto |
| `loading: retired` (tombstone, no adapters) | `intergrax-token-budget`, `intergrax-test-process-budget`, `intergrax-session-artifacts`, `intergrax-git-branch-safety`, `intergrax-no-ai-attribution` |

**Old rule -> new section (W1-P01):** token-budget -> core "Token use"; test-process-budget -> core "Test budget"; session-artifacts -> core "Session artifacts"; git-branch-safety -> core "Branch safety"; no-ai-attribution -> core "Attribution"; env-vendor-qualification -> on-demand skill, core "Environment (C10/C15)" keeps a 2-line note. Worktree rules from `CLAUDE.md` -> skill `igx-worktree`.

**Size gates** (`sync_agent_env.py --check`): L0 (CLAUDE.md + AGENTS.md + always-on rules) <= 8 KB, CLAUDE.md <= 25 lines, AGENTS.md <= 60 lines, descriptions <= 300 chars, skill body <= 120 lines unless listed in `size_exemptions.yaml`.

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
