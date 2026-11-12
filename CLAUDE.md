# Intergrax - Claude Code rules

@AGENTS.md

This file holds what is specific to Claude Code sessions. Rules, skills and plans are shared with Cursor and live in [`agent-env/`](agent-env/README.md) (single source of truth); `.claude/` and `.cursor/` contain only thin adapters that point there.

## Shared agent environment

<!-- BEGIN agent-env:generated (scripts/governance/sync_agent_env.py) - do not edit; edit agent-env/rules/ and re-run -->

### Always-on rules (imported from `agent-env/rules/`)

@agent-env/rules/intergrax-env-vendor-qualification.md
@agent-env/rules/intergrax-git-branch-safety.md
@agent-env/rules/intergrax-session-artifacts.md
@agent-env/rules/intergrax-test-process-budget.md
@agent-env/rules/intergrax-token-budget.md

### On-demand rules (invoke as a skill: `/<name>`; canonical text in `agent-env/rules/`)

| Skill | Invocation | Purpose |
|-------|------------|---------|
| `/intergrax-agent-roles` | operator or model | InterGRAX Delivery DevOS agent roles and handoff routing |
| `/intergrax-ci-hotfix` | operator only | CI/test hotfix - ultra scoped; no docs, no architecture, no full verification |
| `/intergrax-hep-step` | operator only | HEP step execution - strict scoped implementation |
| `/intergrax-independent-audit` | operator or model | Independent G6 audit — implementer must not PASS; separate session only |
| `/intergrax-iteration` | operator only | Intergrax Harness AI - iteration workflow, session protocol (F3) |
| `/intergrax-micro-implement` | operator only | Micro implementation - strict scoped changes for small bounded tasks |
| `/intergrax-scenario-factory` | operator or model | SCENARIO-FACTORY-X (SFX) invariants for the sfx skills and personas - proposal-only drafts under .tmp/sfx/<slug>/ that follow the sfx.*.v1 schemas, secret_ref only, no runtime, no approval or admission claims |

<!-- END agent-env:generated -->

### Vocabulary mapping (shared rules are written in Cursor terms)

| Rule text says | In Claude Code |
|----------------|----------------|
| `@agent-env/rules/<id>.md`, `@<id>` | invoke the `/<id>` skill (or read the file) |
| `Task` / `explore` / `generalPurpose` subagent | `Agent` tool: `Explore` / `general-purpose` |
| `best-of-n-runner` / separate worktree | `Agent` with `isolation: worktree`, or a worktree per the section below |
| `.cursorignore` | not applicable (no indexing); use judgment, read large docs only with `offset`/`limit` |
| "Cursor Settings", Cursor token usage | not applicable |

Plans (plan mode) are written to `agent-env/plans/` (`plansDirectory` in `.claude/settings.json`).

## Git worktrees (agents)

Worktrees live in ONE place **outside the repository**, derived from where the main checkout is - never from a hard-coded path (every developer keeps the project in a different folder).

**Location:** `<parent-of-main-checkout>/_wt/<repo-name>/<slug>`

```bash
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"   # main checkout root, also when run inside a worktree
REPO="$(basename "$MAIN")"                                                       # e.g. intergrax
WT_BASE="$(dirname "$MAIN")/_wt/$REPO"                                           # e.g. <parent>/_wt/intergrax
mkdir -p "$WT_BASE"
git -C "$MAIN" worktree add --detach "$WT_BASE/<slug>" <sha>                     # or: -b <branch> "$WT_BASE/<slug>" <base>
```

- `<slug>`: lowercase `[a-z0-9-]`, max 24 chars, named after the ticket or task (e.g. `t-igx-019-r3`).
- The `_wt` folder is shared by all projects of the developer (one folder per project below it). Create `_wt` and `_wt/<repo-name>` if missing.
- If the parent of the main checkout is itself inside a git repository, or you cannot resolve `MAIN`, STOP and ask the user.
- If `$WT_BASE/<slug>` would exceed ~60 characters (Windows MAX_PATH is 260 and test fixtures nest deeply), shorten the slug or ask.
- NEVER create worktrees inside the repository (`.claude/`, `.tmp/`, any path under `$MAIN`), inside another worktree, or at a drive root.
- Scratch and temporary test directories (for example `pytest --basetemp`) go under `$WT_BASE/_scratch/<slug>`: short paths, outside the repo, create the parent first.

**Working rules**
- Do not search or recurse across worktrees: scope Grep, Glob and `find` to the directory you need; do not run `find`, `du` or `grep -r` over `$MAIN/.claude`, `$MAIN/.tmp` or `_wt`.
- Do not touch, rebase, reset or delete worktrees you did not create. Remove your own with `git worktree remove <path>` and `git worktree prune`; deleting leftover directories is the user's job.
- The main checkout is read-only for parallel agents (it may hold the user's uncommitted work): no pull, reset, stash or checkout there. Work in your own worktree.
- Worktrees that the Claude app creates automatically (`.claude/worktrees/`) are outside this rule's control; leave them as they are and keep them out of searches.
