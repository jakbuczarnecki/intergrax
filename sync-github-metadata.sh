#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Sync GitHub repository description, homepage, and topics from .github/repository-metadata.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'

Usage: ./sync-github-metadata.sh [apply|help]

  ./sync-github-metadata.sh        Validate .github/repository-metadata.json
  ./sync-github-metadata.sh apply  Push description, homepage, and topics to GitHub

Requires: uv. For apply also: gh auth login
EOF
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not installed or not in PATH." >&2
  echo "Install it from: https://docs.astral.sh/uv/" >&2
  exit 1
fi

MODE="${1:-}"

case "$MODE" in
  "")
    echo "[INFO] Dry run - validating manifest only. Use \"apply\" to sync to GitHub."
    uv run python scripts/sync_github_repository_metadata.py
    ;;
  apply)
    if ! command -v gh >/dev/null 2>&1; then
      echo "[ERROR] gh CLI is required for apply. Install: https://cli.github.com/" >&2
      echo "Then authenticate: gh auth login" >&2
      exit 1
    fi
    echo "[INFO] Applying manifest to GitHub repository settings..."
    uv run python scripts/sync_github_repository_metadata.py --apply
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown option: $MODE" >&2
    usage
    exit 1
    ;;
esac
