# Git worktrees (Delivery DevOS)

## Convention

- Branch: `feat/T-IGX-<NN>-<letter>`
- Worktree path: sibling directory `<repo>-wt-T-IGX-<NN>-<letter>` (or operator-defined)
- One active executor per ticket

## Create (Windows)

```powershell
./scripts/governance/worktree/new-worktree.ps1 -TicketId T-IGX-002-A
```

## After checkout

```powershell
uv sync
uv run python -m scripts.governance.igx_gov bootstrap
uv run python -m scripts.governance.igx_gov doctor
```

## Merge back

Operator merges via PR from worktree branch; ticket **Lifecycle state** → `closed` with merge note.

See [`docs/project/technical/devos/PARALLEL-EXECUTION.md`](../../docs/project/technical/devos/PARALLEL-EXECUTION.md).
