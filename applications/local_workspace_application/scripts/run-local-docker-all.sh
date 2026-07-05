#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Start LKW base stack plus every top-level docker-compose.*.yml overlay.
# Internal compose fragments (e.g. sentry.services.yml) are included by overlays
# and are not auto-discovered here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${APP_DIR}/docker"
BASE_COMPOSE="${DOCKER_DIR}/docker-compose.yml"
ENV_FILE="${APP_DIR}/.env"
ENV_EXAMPLE="${APP_DIR}/.env.example"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f "${BASE_COMPOSE}" ]]; then
  echo "Missing base compose file: ${BASE_COMPOSE}" >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_EXAMPLE}" ]]; then
    echo "Missing .env and .env.example in ${APP_DIR}" >&2
    exit 1
  fi
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  echo "Created ${ENV_FILE} from .env.example"
fi

compose_files=("${BASE_COMPOSE}")
while IFS= read -r overlay; do
  compose_files+=("${overlay}")
done < <(find "${DOCKER_DIR}" -maxdepth 1 -name 'docker-compose.*.yml' | sort)

echo "Compose files:"
for file in "${compose_files[@]}"; do
  echo "  ${file}"
done

docker_args=(compose)
for file in "${compose_files[@]}"; do
  docker_args+=(-f "${file}")
done

if [[ $# -eq 0 ]]; then
  docker_args+=(up --build)
else
  docker_args+=("$@")
fi

echo "Running: docker ${docker_args[*]}"
exec docker "${docker_args[@]}"
