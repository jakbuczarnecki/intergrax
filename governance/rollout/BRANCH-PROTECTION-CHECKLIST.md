# Branch protection checklist (Delivery DevOS)

**Scope:** required status checks on `development` — **SERVER** enforcement; not satisfied by repo scripts alone.

## Prerequisites

- `gh auth login` with permission to manage branch protection for the repository.
- Local hooks installed: `uv run python -m scripts.governance.igx_gov bootstrap`.
- Push gate readiness: `uv run python scripts/dev/verify_push_gate_readiness.py`.

## Apply (operator)

```powershell
# From repo root — script path is SSOT
.\scripts\dev\apply_github_branch_protection.ps1
```

Adjust required checks in that script to include at minimum:

- Regression gate / `ci-smoke` (or equivalent smoke job name on default branch workflow)
- `governance-fast` (policy registry + governance unit suite)

Treat `governance-policy-pin` as **ADVISORY** until explicitly added as a required check.

## Verify (mandatory evidence)

```powershell
gh api repos/{owner}/{repo}/branches/development/protection
```

| Outcome | Record |
|---------|--------|
| HTTP 200 + required checks listed | SERVER **PASS** — paste redacted JSON snippet in ticket checkpoint |
| 401/403 or no auth | **EXTERNAL_BLOCKED** — do not claim protection is live |
| 404 | Branch or protection not configured — **BLOCKED** until operator completes step |

## Related

- [`DEVELOPER-QUALITY-PROCESS.md`](../../docs/project/technical/devos/DEVELOPER-QUALITY-PROCESS.md)
- [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md) wave W7
