#!/usr/bin/env bash
# Build Tier-3 application image from monorepo root (Phase N).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PKG="dispute_sim_application"
IMAGE_TAG="${IMAGE_TAG:-dispute_sim-application}"
PORT="8020"

cd "${REPO_ROOT}"

if docker buildx version >/dev/null 2>&1; then
  echo "Building ${IMAGE_TAG} (BuildKit)..."
  docker buildx build \
    -f "applications/${PKG}/docker/Dockerfile" \
    --ignorefile "applications/${PKG}/docker/.dockerignore" \
    -t "${IMAGE_TAG}" \
    .
else
  echo "BuildKit not found — using docker build (consider: docker buildx install)"
  docker build \
    -f "applications/${PKG}/docker/Dockerfile" \
    -t "${IMAGE_TAG}" \
    .
fi

echo ""
echo "Built: ${IMAGE_TAG}"
echo "Run:   docker run --rm --env-file applications/${PKG}/.env -p ${PORT}:${PORT} ${IMAGE_TAG}"
