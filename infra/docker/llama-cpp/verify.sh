#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Start llama.cpp stack (if needed), wait for health, run local-only E2E tests.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_CHAT="$ROOT/infra/docker/llama-cpp/docker-compose.yml"
COMPOSE_EMBED="$ROOT/infra/docker/llama-cpp-embed/docker-compose.yml"
CHAT_URL="${INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL:-http://127.0.0.1:8102/v1}"
EMBED_URL="${INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL:-http://127.0.0.1:8103/v1}"
CHAT_MODELS="${CHAT_URL%/}/models"
EMBED_MODELS="${EMBED_URL%/}/models"
MAX_WAIT_SEC="${LLAMA_CPP_VERIFY_MAX_WAIT_SEC:-900}"
POLL_SEC="${LLAMA_CPP_VERIFY_POLL_SEC:-10}"

export INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL="$CHAT_URL"
export INTERGRAX_DEFAULT_LLAMA_CPP_MODEL="${INTERGRAX_DEFAULT_LLAMA_CPP_MODEL:-default}"
export INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL="$EMBED_URL"
export INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL="${INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL:-default}"
export INTERGRAX_LLAMA_CPP_VERIFY=1

wait_url() {
  local label="$1"
  local url="$2"
  local elapsed=0
  echo "Waiting for $label at $url (max ${MAX_WAIT_SEC}s)..."
  while [ "$elapsed" -lt "$MAX_WAIT_SEC" ]; do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "$label is ready."
      return 0
    fi
    sleep "$POLL_SEC"
    elapsed=$((elapsed + POLL_SEC))
  done
  echo "ERROR: $label not ready after ${MAX_WAIT_SEC}s" >&2
  return 1
}

if ! curl -sf "$CHAT_MODELS" >/dev/null 2>&1; then
  echo "Starting standalone llama.cpp chat + embed containers..."
  docker compose -f "$COMPOSE_CHAT" up -d
  docker compose -f "$COMPOSE_EMBED" up -d
fi

wait_url "llama.cpp chat" "$CHAT_MODELS"
wait_url "llama.cpp embed" "$EMBED_MODELS"

cd "$ROOT"
echo "Running llama.cpp E2E tests (excluded from GitHub CI)..."
uv run pytest tests/e2e/llama_cpp/ -m "e2e and no_ci" -q --tb=short
echo "llama.cpp verify: OK"
