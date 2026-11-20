# Intergrax - Claude Code rules

@AGENTS.md

Rules, skills and plans live in [`agent-env/`](agent-env/README.md) (single source of truth, shared with Cursor); `.claude/` and `.cursor/` hold only generated adapters. Plans go to `agent-env/plans/`. Worktrees: skill `/igx-worktree`. Subagents: `Agent` tool (`igx-researcher`, `igx-implementer`, `igx-reviewer`, `igx-architect`; `general-purpose` is denied); separate worktree = `isolation: worktree`.

<!-- BEGIN agent-env:generated (scripts/governance/sync_agent_env.py) - do not edit; edit agent-env/rules/ and re-run -->
@agent-env/rules/intergrax-core.md
On-demand rules (skills, canonical text in `agent-env/rules/`): `/intergrax-agent-roles`, `/intergrax-ci-hotfix`, `/intergrax-env-vendor-qualification`, `/intergrax-hep-step`, `/intergrax-independent-audit`, `/intergrax-iteration`, `/intergrax-micro-implement`, `/intergrax-scenario-factory`
<!-- END agent-env:generated -->
