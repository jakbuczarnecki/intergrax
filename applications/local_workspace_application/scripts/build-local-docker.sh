#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"
ENV_FILE="$APP_DIR/.env"
ENV_EXAMPLE="$APP_DIR/.env.example"

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
  grep -E "^[[:space:]]*$key[[:space:]]*=" "$ENV_FILE" 2>/dev/null \
    | tail -n 1 \
    | sed -E "s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//" \
    | sed -E 's/^"(.*)"$/\1/' \
    | sed -E "s/^'(.*)'$/\1/" || true
}

MODEL=$(read_env_value "INTERGRAX_DEFAULT_OLLAMA_MODEL")
if [ -z "${MODEL:-}" ]; then
  MODEL=$(read_env_value "INTERGRAX_LLM_MODEL")
fi
if [ -z "${MODEL:-}" ]; then
  MODEL="llama3.1:latest"
fi

cd "$REPO_ROOT"

echo "Building local workspace Docker image..."
docker compose -f "$COMPOSE_FILE" build

echo "Starting Ollama service..."
docker compose -f "$COMPOSE_FILE" up -d ollama

echo "Pulling Ollama model: $MODEL"
for attempt in 1 2 3; do
  if docker compose -f "$COMPOSE_FILE" exec -T ollama ollama pull "$MODEL"; then
    break
  fi
  if [ "$attempt" -eq 3 ]; then
    echo "Ollama pull failed after 3 attempts" >&2
    exit 1
  fi
  echo "Retrying ollama pull ($attempt/3)..."
  sleep 5
done

echo "Starting local stack..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Stack is starting. Verify with:"
echo "  curl http://127.0.0.1:8020/health"
echo "  curl http://127.0.0.1:8020/v1/local_workspace/agents"
