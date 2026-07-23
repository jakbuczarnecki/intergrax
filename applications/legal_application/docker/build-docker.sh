#!/usr/bin/env bash
# Build Tier-3 application image via materialized runtime graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PKG="legal_application"
IMAGE_TAG="${IMAGE_TAG:-legal-application}"
PORT="8000"

cd "${REPO_ROOT}"
uv run python scripts/build/build_application_image.py \
  --application "${PKG}" \
  --tag "${IMAGE_TAG}" \
  --context-dir "applications/${PKG}/docker/runtime-context" \
  --keep-context

echo ""
echo "Built: ${IMAGE_TAG}"
echo "Run:   docker run --rm --env-file applications/${PKG}/.env -p ${PORT}:${PORT} ${IMAGE_TAG}"
