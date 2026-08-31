#!/usr/bin/env bash
# UE-11G-C1 — one-command real agentic production certification
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
uv run python scripts/build/build_application_image.py \
  --application local_workspace_application \
  --context-dir applications/local_workspace_application/docker/runtime-context \
  --materialize-only
docker compose -f tests/system/unified_execution/docker-compose.yml up --build --exit-code-from proof-runner "$@"
