#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

set -e

TOOL="$1"
ACTION="$2"

if [ -z "$TOOL" ] || [ -z "$ACTION" ]; then
  echo "Usage: ./manage.sh <tool> {start|stop|status}"
  exit 1
fi

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$TOOL/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "docker-compose.yml not found for tool '$TOOL'"
  exit 1
fi

case "$ACTION" in
  start)
    echo "Starting $TOOL container..."
    docker compose -f "$COMPOSE_FILE" up -d
    ;;
  stop)
    echo "Stopping $TOOL container..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  status)
    echo "$TOOL container status:"
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  *)
    echo "Usage: ./manage.sh <tool> {start|stop|status}"
    exit 1
    ;;
esac