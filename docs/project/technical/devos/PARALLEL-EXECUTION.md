# Parallel agent execution (no PostgreSQL lease)

InterGRAX uses **procedural fencing** and git worktrees — not a gateway PostgreSQL lease.

## Rules

1. **One ticket = one executor = one worktree** (branch `feat/T-IGX-NN-x`).
2. Second agent on the same ticket → STOP; record `Lease holder:` in ticket or use advisory `ticket_lock.py`.
3. Commit rights are `may_commit` in [`role_registry.yaml`](../../../../governance/agent-control-plane/role_registry.yaml). The implementer may commit on its own worktree branch with `Ref:`. The orchestrator does not commit, merge, push, or open a PR. No subagent closes a batch or declares G6 PASS. The parent reviews `DONE` against git. `igx_gov ticket` is a verification command, not a commit, and it does not grant the orchestrator `may_commit`.

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
