# Intergrax - Claude Code rules

Project instructions live in [`AGENTS.md`](AGENTS.md) (read it first). This file holds rules that are specific to Claude Code sessions.

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
