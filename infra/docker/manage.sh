#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

set -e

TOOL="$1"
ACTION="$2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$TOOL" ] || [ -z "$ACTION" ]; then
  echo "Usage: ./manage.sh <tool|all> {start|stop|status|build}"
  exit 1
fi

build_tool () {

  local TOOL_NAME="$1"
  local TOOL_DIR="$ROOT_DIR/$TOOL_NAME"
  local COMPOSE_FILE="$TOOL_DIR/docker-compose.yml"
  local DOCKERFILE="$TOOL_DIR/Dockerfile"

  if [ ! -f "$DOCKERFILE" ]; then
    echo "Skipping $TOOL_NAME (no Dockerfile)"
    return
  fi

  if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Skipping $TOOL_NAME (no docker-compose.yml)"
    return
  fi

  echo "Building $TOOL_NAME..."
  docker compose -f "$COMPOSE_FILE" build
}

case "$ACTION" in

  build)

    if [ "$TOOL" = "all" ]; then

      for dir in "$ROOT_DIR"/*/ ; do
        TOOL_NAME=$(basename "$dir")
        build_tool "$TOOL_NAME"
      done

    else

      build_tool "$TOOL"

    fi
    ;;

  start)

    COMPOSE_FILE="$ROOT_DIR/$TOOL/docker-compose.yml"

    if [ ! -f "$COMPOSE_FILE" ]; then
      echo "docker-compose.yml not found for tool '$TOOL'"
      exit 1
    fi

    echo "Starting $TOOL container..."
    docker compose -f "$COMPOSE_FILE" up -d
    ;;

  stop)

    COMPOSE_FILE="$ROOT_DIR/$TOOL/docker-compose.yml"

    if [ ! -f "$COMPOSE_FILE" ]; then
      echo "docker-compose.yml not found for tool '$TOOL'"
      exit 1
    fi

    echo "Stopping $TOOL container..."
    docker compose -f "$COMPOSE_FILE" down
    ;;

  status)

    COMPOSE_FILE="$ROOT_DIR/$TOOL/docker-compose.yml"

    if [ ! -f "$COMPOSE_FILE" ]; then
      echo "docker-compose.yml not found for tool '$TOOL'"
      exit 1
    fi

    echo "$TOOL container status:"
    docker compose -f "$COMPOSE_FILE" ps
    ;;

  *)

    echo "Usage: ./manage.sh <tool|all> {start|stop|status|build}"
    exit 1
    ;;

esac