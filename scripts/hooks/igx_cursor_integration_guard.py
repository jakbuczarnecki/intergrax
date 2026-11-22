#!/usr/bin/env python3
"""Cursor-only pre-shell gate for approved remote integration.

This does not replace server-side branch protection. It prevents an agent from
using direct remote mutation commands instead of the OD-AF-INT A runner.
"""
from __future__ import annotations

import json
import re
import sys

# This intentionally denies direct commands for every Cursor role. The
# approved runner invokes git/gh internally after policy validation.
BANNED = (
    re.compile(r"(?i)\bgit(?:\.exe)?\s+push\b"),
    re.compile(r"(?i)\bgh(?:\.exe)?\s+pr\s+(?:create|merge)\b"),
    re.compile(r"(?i)\bgh(?:\.exe)?\s+api\b[^\n]*\s(?:-X|--method)\s*(?:POST|PUT|PATCH|DELETE)\b"),
    re.compile(r"(?i)\bgit(?:\.exe)?\s+reset\s+--hard\b"),
)


def decide(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return {"permission": "deny", "user_message": "Cursor integration hook received invalid shell payload."}
    cmd = value["command"]
    if any(pattern.search(cmd) for pattern in BANNED):
        return {
            "permission": "deny",
            "user_message": "Direct remote mutation is not authorized. Use the OD-AF-INT A scoped integrator runner.",
        }
    return {"permission": "allow"}


def main() -> int:
    try:
        value = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        value = None
    print(json.dumps(decide(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
