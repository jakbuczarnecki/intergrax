#!/usr/bin/env bash
# Create a git worktree for a delivery ticket
set -euo pipefail
TICKET_ID="${1:?Ticket id e.g. T-IGX-002-A}"
BASE_BRANCH="${2:-development}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
SAFE="$(echo "$TICKET_ID" | tr -cd 'A-Za-z0-9-' )"
BRANCH="feat/${SAFE}"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
WT_PATH="${PARENT}/${NAME}-wt-${SAFE}"
git fetch origin "$BASE_BRANCH" 2>/dev/null || true
git worktree add -B "$BRANCH" "$WT_PATH" "origin/${BASE_BRANCH}" || git worktree add -B "$BRANCH" "$WT_PATH" "$BASE_BRANCH"
echo "[INFO] Worktree: $WT_PATH branch $BRANCH"
