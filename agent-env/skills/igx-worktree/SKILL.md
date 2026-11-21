---
name: igx-worktree
description: Create and use git worktrees for agents - the _agent layout, slugs, scratch dirs and the rules for the main checkout
---

# igx-worktree

Worktrees live in ONE place **outside the repository**, never inside it, inside another worktree or at a drive root.

## Layout

- Location: `<parent-of-main-checkout>/_wt/<repo-name>/<slug>`:

```bash
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"   # main checkout root, also inside a worktree
REPO="$(basename "$MAIN")"
WT_BASE="$(dirname "$MAIN")/_wt/$REPO"
mkdir -p "$WT_BASE"
git -C "$MAIN" worktree add --detach "$WT_BASE/<slug>" <sha>                  # or: -b <branch> "$WT_BASE/<slug>" <base>
```

## Rules

- `<slug>`: lowercase `[a-z0-9-]`, max 24 chars, named after the ticket or task (`t-igx-019-r3`).
- Keep the full path under ~60 characters (Windows MAX_PATH is 260 and fixtures nest deeply); shorten the slug.
- If the parent of the main checkout is itself inside a git repository, or `MAIN` cannot be resolved, escalate (missing access) and create nothing.
- Scratch and temporary test directories (for example `pytest --basetemp`) go under `<wt-base>/_scratch/<slug>`: short paths, outside the repo, create the parent first.

## Working rules

- Scope Grep, Glob and `find` to the directory you need. Never recurse across worktrees, `.claude/`, `.tmp/` or `_wt`.
- The main checkout is read-only for parallel agents (it may hold uncommitted work): no pull, reset, stash or checkout there. Work in your own worktree.
- Do not touch, rebase, reset or delete worktrees you did not create. Remove your own with `git worktree remove <path>` and `git worktree prune`.
- Worktrees the Claude app creates itself (`.claude/worktrees/`) are left as they are and kept out of searches.
