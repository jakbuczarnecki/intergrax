#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"

cd "$REPO_ROOT"

echo "Materializing minimal runtime context for governed_contractor_application..."
uv run python scripts/build/build_application_image.py           --application "governed_contractor_application"           --context-dir "applications/governed_contractor_application/docker/runtime-context"           --materialize-only

echo "Building and starting governed contractor via Docker Compose..."
docker compose -f "$COMPOSE_FILE" up --build -d

echo "Stack is starting. Verify with:"
echo "  curl http://127.0.0.1:8000/health"
