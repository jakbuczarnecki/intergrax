# Intergrax pre-push enterprise gate (smoke by default). Install: scripts/dev/install_git_hooks.py
$ErrorActionPreference = "Stop"
$Root = git rev-parse --show-toplevel
Set-Location $Root
& uv run python scripts/ci/run_push_gate.py
exit $LASTEXITCODE
