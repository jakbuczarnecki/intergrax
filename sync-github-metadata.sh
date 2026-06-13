#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Sync GitHub repository description, homepage, and topics from .github/repository-metadata.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'

Usage: ./sync-github-metadata.sh [check|help]

  ./sync-github-metadata.sh        Push description, homepage, and topics to GitHub
  ./sync-github-metadata.sh check  Validate manifest only (dry run)

Requires: uv and gh auth login
EOF
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not installed or not in PATH." >&2
  echo "Install it from: https://docs.astral.sh/uv/" >&2
  exit 1
fi

MODE="${1:-}"

case "$MODE" in
  ""|apply)
    if ! command -v gh >/dev/null 2>&1; then
      echo "[ERROR] gh CLI is required. Install: https://cli.github.com/" >&2
      echo "Then authenticate: gh auth login" >&2
      exit 1
    fi
    echo "[INFO] Applying manifest to GitHub repository settings..."
    uv run python scripts/sync_github_repository_metadata.py --apply
    ;;
  check|dry-run|validate)
    echo "[INFO] Dry run - validating manifest only (no GitHub changes)."
    uv run python scripts/sync_github_repository_metadata.py
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
