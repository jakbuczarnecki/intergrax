#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

set -e

ACTION="$1"
COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "docker-compose.yml not found in $(dirname "$COMPOSE_FILE")"
  exit 1
fi

case "$ACTION" in
  start)
    echo "Starting Integration stack..."
    docker compose -f "$COMPOSE_FILE" up -d
    ;;
  stop)
    echo "Stopping Integration stack..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  status)
    echo "Integration stack status:"
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  *)
    echo "Usage: ./manage.sh {start|stop|status}"
    exit 1
    ;;
esac