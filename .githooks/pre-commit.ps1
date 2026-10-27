$ErrorActionPreference = "Stop"
$Root = git rev-parse --show-toplevel
Set-Location $Root
& uv run python -u scripts/ci/run_pre_commit_gate.py
exit $LASTEXITCODE
