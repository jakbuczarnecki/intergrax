#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.

set -e

ACTION="${1:-}"
PROFILE="${2:-default}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT/docker-compose.yml"

PROFILE_FLAGS=()

add_profiles() {
  for p in "$@"; do
    PROFILE_FLAGS+=(--profile "$p")
  done
}

resolve_profiles() {
  case "$PROFILE" in
    all)
      add_profiles core queue rag data secrets observability cloud heavy vllm llama-cpp all
      ;;
    core|queue|rag|data|secrets|observability|cloud|heavy|vllm|llama-cpp)
      add_profiles "$PROFILE"
      ;;
    p6)
      add_profiles core p6
      ;;
    default)
      add_profiles core queue rag data secrets
      ;;
    minimal)
      add_profiles core
      ;;
    *)
      echo "Unknown profile: $PROFILE"
      echo "Usage: ./manage.sh {start|stop|status|build} [profile]"
      echo "Profiles: default, minimal, core, queue, rag, data, secrets, observability, cloud, heavy, vllm, llama-cpp, p6, all"
      exit 1
      ;;
  esac
}

if [ -z "$ACTION" ]; then
  echo "Usage: ./manage.sh {start|stop|status|build} [profile]"
  exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "docker-compose.yml not found in $ROOT"
  exit 1
fi

resolve_profiles

case "$ACTION" in
  start)
    echo "Starting Integration stack (profile=$PROFILE)..."
    docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" up -d
    ;;
  stop)
    echo "Stopping Integration stack (profile=$PROFILE)..."
    docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" down
    ;;
  status)
    echo "Integration stack status (profile=$PROFILE):"
    docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" ps -a
    ;;
  build)
    echo "Building custom images (docling)..."
    docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" build docling
    ;;
  *)
    echo "Usage: ./manage.sh {start|stop|status|build} [profile]"
    exit 1
    ;;
esac
