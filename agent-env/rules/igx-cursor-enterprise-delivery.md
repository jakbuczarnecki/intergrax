---
description: Cursor enterprise autonomous delivery on single checkout — git, venv, publish
loading: explicit
---

# Cursor enterprise autonomous delivery

Operator-approved delivery batches on the **main checkout** (`development`) follow **implement → test → commit → integrate → publish → verify** without delegating routine Git to the operator.

## Execution surface

- **Checkout:** repository root on branch `development` (no extra worktree unless the operator explicitly authorizes `/igx-worktree` or Software Factory).
- **Commits:** scoped `git add <paths>` only; hooks always on; trailer `Ref: governance/tickets/T-IGX-<NN>-<L>.md`; no AI-tool attribution.
- **Push:** non-force `git push origin development` when branch protection and the approved mandate allow it; otherwise a short-lived publish branch in the same checkout + `gh pr create` (no worktree for PR only).
- **Supersedes:** in Cursor direct execution, commit/push on `development` within approved ticket scope follows [`igx-cursor-direct-execution.md`](igx-cursor-direct-execution.md), not the orchestrator `may_commit: false` row in `role_registry.yaml` (delegated workers keep registry rules).

## Python environments

| Tier | Path | Use |
|------|------|-----|
| Canonical | `<repo>/.venv` | `igx_gov ticket`, pre-commit, governance conformance |
| Qualification | `.tmp/session/<slug>/.venv` via `UV_PROJECT_ENVIRONMENT` | parallel or long test runs that must not write the canonical tree |

Rules:

- Before a campaign: one `uv sync --extra dev-ci --extra dev-ci-rag --extra dev-unit-cert --frozen` on the canonical `.venv` only when no other process holds extension modules (see `scripts/cursor/igx_canonical_venv_status.py`).
- During qualification: `uv run --no-sync` with the intended interpreter; do not run `uv sync` mid-run.
- Pre-commit and CI ruff gates use `uv run --no-sync` so hooks never mutate the environment during another session's work.

## Parallelism

- Read-only subagents may run in parallel; at most one writer on the main checkout.
- Serialize pytest on the canonical venv (`.tmp/factory/pytest.lock` via `scripts/cursor/igx_test_lock.py`).
- Do not kill another session's Python without operator authorization; coordinate or use an isolated qualification venv.

## Integration

- Diverged `development`: merge `origin/development` (no unauthorized rebase/reset/force-push).
- Foreign-owner content conflicts: escalate with paths and owners.
- Untracked files that differ from `origin`: preserve until owner disposition (no blind delete).

## Publication verify

After push or PR: record remote SHA or PR URL and required CI state. Merge and branch protection overrides stay human when hooks or rulesets require it.