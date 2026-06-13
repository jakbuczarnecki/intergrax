#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Sync GitHub repository metadata — see README.md in this folder.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SCRIPT=".github/repo-management/sync_github_repository_metadata.py"

usage() {
  cat <<'EOF'

Usage: ./.github/repo-management/sync-github-metadata.sh [check|help]

  sync-github-metadata.sh        Push description, homepage, and topics to GitHub
  sync-github-metadata.sh check  Validate manifest only (dry run)

Setup: .github/repo-management/README.md
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
    echo "[INFO] Applying manifest to GitHub repository settings..."
    uv run python "$SCRIPT" --apply
    ;;
  check|dry-run|validate)
    echo "[INFO] Dry run - validating manifest only (no GitHub changes)."
    uv run python "$SCRIPT"
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
