#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"
COMPOSE_PROJECT_NAME="intergrax_lkw"

cd "$REPO_ROOT"

echo "Materializing minimal runtime context for local_workspace_application..."
uv run python scripts/build/build_application_image.py           --application "local_workspace_application"           --context-dir "applications/local_workspace_application/docker/runtime-context"           --materialize-only

echo "Building and starting local workspace via Docker Compose..."
docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" up --build -d

echo "Ensuring the configured default Ollama model is available..."
docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" exec -T ollama ollama pull llama3.1:latest

echo "Stack is starting. Verify with:"
echo "  curl http://127.0.0.1:8020/health"
