#!/usr/bin/env bash
# Build Tier-3 application image from monorepo root (Phase N).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PKG="attestation_demo"
IMAGE_TAG="${IMAGE_TAG:-attestation-demo}"
PORT="8097"
DOCKERFILE="applications/${PKG}/docker/Dockerfile"

cd "${REPO_ROOT}"

classic_build() {
  echo "Building ${IMAGE_TAG} (classic docker build)..."
  docker build -f "${DOCKERFILE}" -t "${IMAGE_TAG}" .
}

if docker buildx version >/dev/null 2>&1; then
  echo "Building ${IMAGE_TAG} (BuildKit)..."
  if docker buildx build \
    -f "${DOCKERFILE}" \
    --ignorefile "applications/${PKG}/docker/.dockerignore" \
    -t "${IMAGE_TAG}" \
    --load \
    .; then
    :
  else
    echo "BuildKit build failed (often --ignorefile on older buildx) — falling back to classic docker build."
    classic_build
  fi
else
  echo "BuildKit not found — using docker build (consider: docker buildx install)"
  classic_build
fi

echo ""
echo "Built: ${IMAGE_TAG}"
echo "Run:   docker run --rm --env-file applications/${PKG}/.env -p ${PORT}:${PORT} ${IMAGE_TAG}"
