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

1. `uv sync --extra dev-ci --frozen` (skipped in CI and pre-push when `skip_uv_sync_on_push`)
2. `check_environment_conformance.py --profile ci`
3. `uv lock --check`
4. `run_ci_smoke_pytest.py` (`ci_smoke` marker)
5. `check_harness_no_getattr.py --smoke` (agents tree; full scan in `enterprise` profile)
6. `check_agents_no_tier3_imports.py`

Manual equivalent (hook parity):

```bash
uv run python -u scripts/ci/run_push_gate.py
```

Full smoke including `uv sync` (CI parity with fresh env):

```bash
uv run python scripts/ci/run_regression_gate_ci.py --profile smoke
```

Windows batch wrapper: `scripts/ci/run_before_push_ci_smoke.bat`.

## Configuration (`pyproject.toml`)

```toml
[tool.intergrax.push_gate]
profile = "smoke"   # smoke | full | all | off
skip_env = "INTERGRAX_SKIP_PUSH_GATE"
skip_uv_sync_on_push = true
```

- **smoke** — default; blocks push on failure.
- **skip_uv_sync_on_push** — pre-push smoke skips `uv sync` (faster; avoids venv churn). After `uv.lock` changes run `uv sync --extra dev-ci --frozen` once.
- **full** / **all** — long-running; for maintainers only.
- **off** — hook installed but gate disabled locally.

Pre-push entrypoint: `scripts/ci/run_push_gate.py` (not the raw regression runner).

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
- **Branch protection (recommended):** GitHub → Settings → Branches → rule on `development` (and `main`) → require status check **Regression gate** / job `ci-smoke` before merge.
- Ensure the workflow file on the repository **default branch** matches `development` (PR checks use default-branch workflow definition).

## Rollout readiness

```bash
uv run python scripts/dev/verify_push_gate_readiness.py
```

Per developer: `install_git_hooks.py` → `verify_push_gate_readiness.py` → one trial `run_push_gate.py` before first push.

### Windows note

Pre-push runs subprocesses without `cmd.exe` `&&` chains (avoids hangs after long pytest runs). Logs use unbuffered Python (`-u`).
