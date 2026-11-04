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

## Closure program

Remaining DCP work is sequenced in [`governance/DCP-CLOSURE-BACKLOG.md`](../../../governance/DCP-CLOSURE-BACKLOG.md) (waves W1–W8).

When executing closure waves:

- Prefer **orchestrator** + parallel `explore` / `generalPurpose` for read-only gap analysis across governance vs CI vs docs trees.
- Use **best-of-n-runner** or a dedicated worktree per ticket (`T-IGX-005-A` … `008-A`); do not parallelize edits to the same manifest or registry file.
- One `uv run pytest` (or `igx_gov doctor`) per verification step; parent merges and commits with `Ref:`.
- W7 (branch protection) and W8 (G6) are operator / independent-session boundaries — no subagent may declare PASS or flip manifest certification flags.
