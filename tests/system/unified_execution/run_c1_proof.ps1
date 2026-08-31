# UE-11G-C1 — one-command real agentic production certification
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
uv run python scripts/build/build_application_image.py `
  --application local_workspace_application `
  --context-dir applications/local_workspace_application/docker/runtime-context `
  --materialize-only
docker compose -f tests/system/unified_execution/docker-compose.yml up --build --exit-code-from proof-runner @args
