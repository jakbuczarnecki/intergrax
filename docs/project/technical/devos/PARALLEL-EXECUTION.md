# Parallel agent execution (no PostgreSQL lease)

InterGRAX uses **procedural fencing** and git worktrees — not a gateway PostgreSQL lease.

## Rules

1. **One ticket = one executor = one worktree** (branch `feat/T-IGX-NN-x`).
2. Second agent on the same ticket → STOP; record `Lease holder:` in ticket or use advisory `ticket_lock.py`.
3. Subagents **must not** commit, close batches, or declare G6 PASS — parent synthesizes and runs `igx_gov ticket`.

## Decision tree

| Need | Mechanism |
|------|-----------|
| Read-only search | `explore` / `generalPurpose` (max 2–3 parallel) |
| Isolated code tries | `best-of-n-runner` or new worktree |
| Single change set | One implementer in ticket scope |

## Worktree bootstrap

See [`governance/workflows/WORKTREES.md`](../../../../governance/workflows/WORKTREES.md) and `scripts/governance/worktree/new-worktree.ps1`.

After create: `uv sync`, `igx_gov bootstrap`, `igx_gov doctor`.

## Skill

 `@igx-devos-parallel-orchestration`
