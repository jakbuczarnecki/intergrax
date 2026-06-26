#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"

cd "$REPO_ROOT"

echo "Building and starting intergrax assistant via Docker Compose..."
docker compose -f "$COMPOSE_FILE" up --build -d

echo "Stack is starting. Verify with:"
echo "  curl http://127.0.0.1:8040/health"
