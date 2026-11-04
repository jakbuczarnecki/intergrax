---
name: igx-devos-parallel-orchestration
description: When to use Task subagents vs worktrees for parallel InterGRAX delivery work.
---

# Parallel orchestration

| Situation | Action |
|-----------|--------|
| Read-only exploration | `explore` / `generalPurpose`, max 2–3 parallel |
| Isolated code attempts | `best-of-n-runner` or separate git worktree |
| Single batch | One agent in ticket scope |

**Prohibited for subagents:** commit, declare done, G6 PASS. Parent runs `igx_gov ticket` and owns `Ref:`.

Optional advisory lock: `uv run python scripts/governance/ticket_lock.py acquire T-IGX-NNN-L --holder <session>`.

See `docs/project/technical/devos/PARALLEL-EXECUTION.md`.
