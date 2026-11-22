#!/usr/bin/env python3
"""Cursor-only pre-shell gate for approved remote integration.

This does not replace server-side branch protection. It prevents an agent from
using direct remote mutation or destructive ref commands instead of the
OD-AF-INT A runner. Global options between the executable and the subcommand
(``git -C <path> push``, ``git --git-dir=x push``, ``git.exe -c k=v push``) are
covered. Known residual gaps: git aliases, scripts that call git internally and
shell indirection (variables, command substitution).
"""

from __future__ import annotations

import json
import re
import sys

# One option value: quoted string or bare word up to a shell separator. A value
# never starts with "-", which keeps option parsing unambiguous (no exponential
# backtracking on long flag lists).
_VALUE = r"""(?:"[^"]*"|'[^']*'|[^\s;&|"'-][^\s;&|]*)"""
# Any number of options, each with an optional value (-C <path>, -c k=v, --no-pager).
_OPTS = rf"(?:\s+-[^\s;&|]*(?:\s+{_VALUE})?)*"
# Executable name is case-insensitive (Windows); an exe path may end with a quote.
_GIT = rf"""\b(?i:git(?:\.exe|\.cmd)?)["']?{_OPTS}"""
_GH = rf"""\b(?i:gh(?:\.exe)?)["']?{_OPTS}"""
# Rest of the current shell segment (stops at ; & | and newlines).
_SEG = r"[^;&|\n]*"


def _sub(names: str) -> str:
    """Subcommand token, optionally quoted, case-insensitive."""
    return rf"""\s+["']?(?i:{names})\b["']?"""


# Flags are matched case-sensitively (-D force-deletes, -d does not).
_FORCE = r"\s(?:--force\b|-[A-Za-z]*f)"

# This intentionally denies direct commands for every Cursor role. The
# approved runner invokes git/gh internally after policy validation.
BANNED = (
    re.compile(_GIT + _sub("push|send-pack|http-push")),
    re.compile(_GIT + _sub("subtree") + _SEG + r"\spush\b"),
    re.compile(_GIT + _sub("reset") + _SEG + r"\s--hard\b"),
    re.compile(_GIT + _sub("branch") + rf"(?={_SEG}\s-[A-Za-z]*D)"),
    re.compile(_GIT + _sub("branch") + rf"(?={_SEG}\s(?:--delete\b|-[A-Za-z]*d))(?={_SEG}{_FORCE})"),
    re.compile(_GIT + _sub("worktree") + r"\s+(?i:remove)\b" + rf"(?={_SEG}{_FORCE})"),
    re.compile(_GIT + _sub("update-ref") + rf"(?={_SEG}\s(?:-d\b|--delete\b))"),
    re.compile(_GH + _sub("pr") + _OPTS + _sub("create|new|merge")),
    re.compile(_GH + _sub("api") + rf"""(?={_SEG}(?:\s-X\s*|\s--method(?:\s+|=))["']?(?i:POST|PUT|PATCH|DELETE)\b)"""),
)

_GH_API_TAIL = re.compile(_GH + _sub("api") + rf"({_SEG})")
# Field/input flags make `gh api` default to POST.
_FIELD = re.compile(r"\s(?:-[fF]|--field\b|--raw-field\b|--input\b)")
_EXPLICIT_READ = re.compile(r"""(?:\s-X\s*|\s--method(?:\s+|=))["']?(?i:GET|HEAD)\b""")


def _gh_api_implicit_post(cmd: str) -> bool:
    for match in _GH_API_TAIL.finditer(cmd):
        tail = match.group(1)
        if not _FIELD.search(tail) or _EXPLICIT_READ.search(tail):
            continue
        is_graphql = re.match(r"""\s*["']?graphql\b""", tail) is not None
        # A graphql read is allowed only when the query is visible inline.
        if is_graphql and not re.search(r"(?i)\bmutation\b|--input\b|=@", tail):
            continue
        return True
    return False


def _normalize(cmd: str) -> str:
    # Shell line continuations (PowerShell backtick, POSIX backslash) join lines.
    return re.sub(r"(?:`|\\)\r?\n", " ", cmd)


def decide(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return {"permission": "deny", "user_message": "Cursor integration hook received invalid shell payload."}
    cmd = _normalize(value["command"])
    if any(pattern.search(cmd) for pattern in BANNED) or _gh_api_implicit_post(cmd):
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
