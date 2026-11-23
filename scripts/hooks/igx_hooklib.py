# © Artur Czarnecki. All rights reserved.
"""Shared helpers for the igx PreToolUse guards (stdlib only)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

MAX_BYTES = 80 * 1024
MAX_LINES = 1500


def disabled() -> bool:
    """Kill switch: ``IGX_HOOKS=off`` disables only the igx read and bash guards."""
    return os.environ.get("IGX_HOOKS", "").strip().lower() == "off"


def load_input() -> dict[str, Any]:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def deny(reason: str) -> int:
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    sys.stdout.write(json.dumps(out))
    return 0


def state_path(session_id: str) -> Path:
    safe = "".join(c for c in (session_id or "nosession") if c.isalnum() or c in "-_")[:64] or "nosession"
    base = Path(os.environ.get("IGX_HOOK_STATE_DIR") or Path(tempfile.gettempdir()) / "igx-read-guard")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{safe}.json"


def load_state(session_id: str) -> dict[str, Any]:
    try:
        data = json.loads(state_path(session_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(session_id: str, state: dict[str, Any]) -> None:
    path = state_path(session_id)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def count_lines(path: Path, limit: int = MAX_LINES + 1) -> int:
    n = 0
    try:
        with path.open("rb") as fh:
            for _ in fh:
                n += 1
                if n >= limit:
                    break
    except OSError:
        return 0
    return n
