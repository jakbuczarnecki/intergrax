# Enterprise quality workflow (mechanical gates)

Operator-facing map of **automated** quality enforcement. Semantic **FRZ-*** freeze acceptance remains in [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](../../../maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md); mechanical gate inventory in [QUAL-X](../../../maintainers/qualification/QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md).

## Layers

| Layer | When | Entrypoint |
|-------|------|------------|
| Pre-commit | `git commit` | `.githooks/pre-commit` → `run_pre_commit_gate.py` (ruff + diff-scoped) |
| Pre-push | `git push` | `.githooks/pre-push` → `run_push_gate.py` (diff-scoped + smoke) |
| CI smoke | PR / push `main`, `development` | `.github/workflows/unit-tests.yml` |
| Enterprise local | Maintainer | `run_regression_gate_ci.py --profile enterprise` |
| Full regression | Dispatch / nightly parity | `--profile full` / `all` |

Configuration: `[tool.intergrax.push_gate]` in `pyproject.toml`.

## Install (each clone)

```bash
uv run python scripts/dev/install_git_hooks.py
uv run python scripts/dev/verify_push_gate_readiness.py
```

## Profiles

- **smoke** — default pre-push; SSOT with CI `ci-smoke`.
- **enterprise** — smoke + vendor/docs/production governance scripts (QUAL-X subset).
- **full** / **all** — full unit gate + governance wiring + closeout (long).

## Diff-scoped gates (TOOL-ROI-1.5 subset)

`scripts/ci/diff_scoped_gates.py` runs heavy maintenance scripts only when changed paths match scope (agents, harness scan roots, applications, docs pairs). Used in pre-commit (staged) and pre-push (commits since merge-base).

## Server enforcement

1. CI on `development` / `main` (workflow in repo).
2. Branch protection (operator, admin): `scripts/dev/apply_github_branch_protection.ps1` after `gh auth login`.

## Bypass (emergency only)

- Pre-push: `INTERGRAX_SKIP_PUSH_GATE=1`
- Pre-commit: `INTERGRAX_SKIP_PRE_COMMIT_GATE=1`

See also [LOCAL_PUSH_GATE.md](LOCAL_PUSH_GATE.md).
