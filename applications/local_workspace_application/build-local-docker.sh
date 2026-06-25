# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
COMPOSE_FILE="$SCRIPT_DIR/docker/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

if [ ! -f "$ENV_FILE" ]; then
  if [ ! -f "$ENV_EXAMPLE" ]; then
    echo "Missing .env and .env.example in $SCRIPT_DIR" >&2
    exit 1
  fi
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  echo "Created $ENV_FILE from .env.example"
fi

read_env_value() {
  key="$1"
  if [ ! -f "$ENV_FILE" ]; then
    return 0
  fi
  grep -E "^[[:space:]]*$key[[:space:]]*=" "$ENV_FILE" \
    | tail -n 1 \
    | sed -E "s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//" \
    | sed -E 's/^"(.*)"$/\1/' \
    | sed -E "s/^'(.*)'$/\1/"
}

MODEL=$(read_env_value "INTERGRAX_DEFAULT_OLLAMA_MODEL" || true)
if [ -z "${MODEL:-}" ]; then
  MODEL=$(read_env_value "INTERGRAX_LLM_MODEL" || true)
fi
if [ -z "${MODEL:-}" ]; then
  MODEL="llama3.1:latest"
fi

cd "$REPO_ROOT"

echo "Building LKW Docker image and local services..."
docker compose -f "$COMPOSE_FILE" build

echo "Starting Ollama service..."
docker compose -f "$COMPOSE_FILE" up -d ollama

echo "Pulling Ollama model: $MODEL"
docker compose -f "$COMPOSE_FILE" exec ollama ollama pull "$MODEL"

echo "Starting LKW local stack..."
docker compose -f "$COMPOSE_FILE" up -d

echo "LKW stack is starting. Verify with:"
echo "  curl http://127.0.0.1:8020/health"
echo "  curl http://127.0.0.1:8020/v1/local_workspace/agents"
