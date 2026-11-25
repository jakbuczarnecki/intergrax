# P0 local replay — ARCH-FREEZE unpublished paths

| Field | Value |
| --- | --- |
| Base | `c7c11d3aa98615c0ecc9c41d6bb67035af302b16` (`arch-freeze/fc-train` / af-fc-p0 P0 commit) |
| Branch / worktree | `af-fc/local-replay` @ `D:/Projekty/_wt/intergrax/af-fc-replay` |
| Source diffs | read-only main checkout `D:/Projekty/intergrax` — `git diff HEAD -- <path>` per disposition list |
| Date | 2026-11-25 |

## Counts

| Result | Count |
| --- | ---: |
| Applied (clean `git apply`) | 65 |
| Skipped (apply conflict) | 0 |
| Foreign paths left untouched | 100 |

Conflict paths: none.

## Note

Disposition list from `P0_DIRTY_TREE_DISPOSITION.md` / `_scratch/af-fc-p0/arch_freeze_paths.txt`. Diffs taken as `git diff HEAD -- <path>` (staged + unstaged on main). Foreign list not replayed. Main checkout unchanged. Commit hook applied ruff format on four replayed Python files and one `E402` noqa on `check_harness_observability_wiring.py` (sys.path bootstrap).