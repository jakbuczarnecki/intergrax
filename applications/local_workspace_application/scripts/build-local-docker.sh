#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_DIR/../.." && pwd)
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"
COMPOSE_PROJECT_NAME="intergrax_lkw"

cd "$REPO_ROOT"

if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

if [ -z "${INTERGRAX_DEFAULT_OLLAMA_MODEL:-}" ]; then
  INTERGRAX_DEFAULT_OLLAMA_MODEL=$(
    awk -v host_llm="${INTERGRAX_LLM_MODEL:-}" -F= '
        function trim(value) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
          if (value ~ /^".*"$/ || value ~ /^'\''.*'\''$/) {
            value = substr(value, 2, length(value) - 2)
          }
          return value
        }
        {
          key = $1
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
          value = $0
          sub(/^[^=]*=/, "", value)
          value = trim(value)
          if (key == "INTERGRAX_DEFAULT_OLLAMA_MODEL") {
            default_model = value
          } else if (key == "INTERGRAX_LLM_MODEL") {
            llm_model = value
          }
        }
        END {
          if (default_model != "") print default_model
          else if (host_llm != "") print host_llm
          else if (llm_model != "") print llm_model
          else print "llama3.1:latest"
        }
      ' "$APP_DIR/.env"
  )
fi

case "$INTERGRAX_DEFAULT_OLLAMA_MODEL" in
  ''|*[!A-Za-z0-9._:/-]*)
    echo "Invalid supported generation-model configuration." >&2
    exit 1
    ;;
esac
export INTERGRAX_DEFAULT_OLLAMA_MODEL

echo "Materializing minimal runtime context for local_workspace_application..."
uv run python scripts/build/build_application_image.py           --application "local_workspace_application"           --context-dir "applications/local_workspace_application/docker/runtime-context"           --materialize-only

echo "Building and starting local workspace via Docker Compose..."
docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" up --build -d

echo "Ensuring the configured Ollama generation model is available..."
docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" exec -T \
  --env "INTERGRAX_DEFAULT_OLLAMA_MODEL=$INTERGRAX_DEFAULT_OLLAMA_MODEL" \
  ollama sh -c 'ollama pull "$INTERGRAX_DEFAULT_OLLAMA_MODEL"'

echo "Stack is starting. Verify with:"
echo "  curl http://127.0.0.1:8020/health"
