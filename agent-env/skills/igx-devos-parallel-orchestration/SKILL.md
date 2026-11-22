---
name: igx-devos-parallel-orchestration
description: When to use Task subagents vs worktrees for parallel InterGRAX delivery work.
---

# Parallel orchestration

| Situation | Action |
|-----------|--------|
| Read-only exploration | `Explore` / `igx-researcher`, max 2–3 parallel |
| Isolated code attempts | `best-of-n-runner` or separate git worktree |
| Single batch | One agent in ticket scope |

**Commit SSOT:** `may_commit` in `governance/agent-control-plane/role_registry.yaml`. An `igx-implementer` subagent may commit on its own worktree branch with `Ref:`. The orchestrator may not commit, merge, push, or open a PR. No subagent declares G6 PASS or closes the batch. The parent reviews `DONE` against git.

Optional advisory lock: `uv run python scripts/governance/ticket_lock.py acquire T-IGX-NNN-L --holder <session>`.

See `docs/project/technical/devos/PARALLEL-EXECUTION.md`.

## Closure program

Remaining DCP work is sequenced in [`governance/DCP-CLOSURE-BACKLOG.md`](../../../governance/DCP-CLOSURE-BACKLOG.md) (waves W1–W8).

When executing closure waves:

- Prefer **orchestrator** + parallel `Explore` / `igx-researcher` for read-only gap analysis across governance vs CI vs docs trees.
- Use **best-of-n-runner** or a dedicated worktree per ticket (`T-IGX-005-A` … `008-A`); do not parallelize edits to the same manifest or registry file.
- One `uv run pytest` (or `igx_gov doctor`) per verification step. Do not run those in parallel.
- `igx-implementer` and `best-of-n-runner` stay at `max_parallel: 2`. A 3–4 worker target is not authorized until a resource benchmark is recorded. Do not raise the cap in this file.
- W7 (branch protection) and W8 (G6) are operator / independent-session boundaries — no subagent may declare PASS or flip manifest certification flags.
