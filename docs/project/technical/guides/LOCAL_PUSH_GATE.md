# Local pre-push gate (smoke)

Mechanical regression checks before `git push` — parity with the GitHub **Regression gate** `ci-smoke` job. This is **not** a full [FRZ-*](../../../maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) enterprise-freeze certification (those criteria need qualification SHA, audits, and evidence ledgers). It enforces the **automated gates** mapped in [QUAL-X gate inventory](../../../maintainers/qualification/QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md).

## Install (once per clone)

```bash
uv run python scripts/dev/install_git_hooks.py
```

Verify:

```bash
uv run python scripts/dev/install_git_hooks.py --status
```

Unset:

```bash
uv run python scripts/dev/install_git_hooks.py --uninstall
```

Git sets `core.hooksPath` to `.githooks` **only in this repository** (not global git config).

On Windows, Git for Windows runs `.githooks/pre-push` via `sh`. You can also run `.githooks/pre-push.ps1` manually.

## What smoke runs (SSOT)

Implemented in `scripts/ci/run_regression_gate_ci.py` → `smoke_steps()`:

1. `uv sync --extra dev-ci --frozen` (skipped in CI via `--skip-uv-sync`)
2. `check_environment_conformance.py --profile ci`
3. `uv lock --check`
4. `run_ci_smoke_pytest.py` (`ci_smoke` marker)
5. Tier boundary: `check_harness_no_getattr.py`, `check_agents_no_tier3_imports.py`

Manual equivalent:

```bash
uv run python scripts/ci/run_regression_gate_ci.py --profile smoke
```

Windows batch wrapper: `scripts/ci/run_before_push_ci_smoke.bat`.

## Configuration (`pyproject.toml`)

```toml
[tool.intergrax.push_gate]
profile = "smoke"   # smoke | full | all | off
skip_env = "INTERGRAX_SKIP_PUSH_GATE"
```

- **smoke** — default; blocks push on failure.
- **full** / **all** — long-running; for maintainers only.
- **off** — hook installed but gate disabled locally.

## Emergency bypass

Fail-closed by default. To skip **one** push (emergencies only):

```bash
# PowerShell
$env:INTERGRAX_SKIP_PUSH_GATE = "1"; git push

# bash
INTERGRAX_SKIP_PUSH_GATE=1 git push
```

Do not commit secrets; gate logs must stay free of credentials.

## Server backup

- **Pull requests** and **push to `main` / `development`** run the same smoke SSOT in [`.github/workflows/unit-tests.yml`](../../../../.github/workflows/unit-tests.yml).
- Feature-branch pushes without an open PR may rely on the local hook until a PR is opened.
- Optional: branch protection requiring the **Regression gate** check on merge to `development`.
