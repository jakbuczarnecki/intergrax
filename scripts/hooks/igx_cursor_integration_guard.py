#!/usr/bin/env python3
"""Cursor-only pre-shell gate for git/gh commands.

Allows a guarded ``git push origin development`` from branch ``development`` when
``cursor_development_push`` preflight passes. Denies force push, other refs,
destructive git operations and direct ``gh`` remote mutations. Does not bypass
Git pre-push hooks or GitHub branch protection.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

_PUSH_MOD = Path(__file__).resolve().parent / "cursor_development_push.py"
_spec = importlib.util.spec_from_file_location("cursor_development_push", _PUSH_MOD)
assert _spec and _spec.loader
_push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_push)
push_target_is_allowed_development = _push.push_target_is_allowed_development
validate_development_push = _push.validate_development_push

_REPO = Path(__file__).resolve().parents[2]
_VALUE = r"""(?:"[^"]*"|'[^']*'|[^\s;&|"'-][^\s;&|]*)"""
_OPTS = rf"(?:\s+-[^\s;&|]*(?:\s+{_VALUE})?)*"
_GIT = rf"""\b(?i:git(?:\.exe|\.cmd)?)["']?{_OPTS}"""
_GH = rf"""\b(?i:gh(?:\.exe)?)["']?{_OPTS}"""
_SEG = r"[^;&|\n]*"


def _sub(names: str) -> str:
    return rf"""\s+["']?(?i:{names})\b["']?"""


_FORCE = r"\s(?:--force\b|-[A-Za-z]*f)"
_PUSH_CMD = re.compile(_GIT + _sub("push"))
_PUSH_FORCE = re.compile(r"(?i)(?:^|\s)(?:\+|--force\b|--force-with-lease\b)")

BANNED = (
    re.compile(_GIT + _sub("send-pack|http-push")),
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
_FIELD = re.compile(r"\s(?:-[fF]|--field\b|--raw-field\b|--input\b)")
_EXPLICIT_READ = re.compile(r"""(?:\s-X\s*|\s--method(?:\s+|=))["']?(?i:GET|HEAD)\b""")
_REPO_FLAG = re.compile(rf"""(?i:(?:^|[\s;&|])(?:git(?:\.exe|\.cmd)?["']?{_OPTS}\s+)?)-[Cc]\s+({_VALUE})""")


def _gh_api_implicit_post(cmd: str) -> bool:
    for match in _GH_API_TAIL.finditer(cmd):
        tail = match.group(1)
        if not _FIELD.search(tail) or _EXPLICIT_READ.search(tail):
            continue
        is_graphql = re.match(r"""\s*["']?graphql\b""", tail) is not None
        if is_graphql and not re.search(r"(?i)\bmutation\b|--input\b|=@", tail):
            continue
        return True
    return False


def _normalize(cmd: str) -> str:
    return re.sub(r"(?:`|\\)\r?\n", " ", cmd)


def _strip_quotes(value: str) -> str:
    text = value.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def _repo_root(cmd: str) -> Path:
    match = _REPO_FLAG.search(cmd)
    if match:
        return Path(_strip_quotes(match.group(1)))
    return _REPO


def _push_denial(cmd: str) -> dict[str, str] | None:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return None
    for segment in re.split(r"[;&|]", cmd):
        piece = segment.strip()
        if not piece:
            continue
        match = _PUSH_CMD.search(piece)
        if not match:
            continue
        tail = piece[match.end() :].strip()
        if _PUSH_FORCE.search(tail):
            return {
                "permission": "deny",
                "user_message": "Force push is not allowed. Publish only a fast-forward to origin/development.",
            }
        if not push_target_is_allowed_development(tail):
            return {
                "permission": "deny",
                "user_message": (
                    "Only non-force `git push origin development` (or HEAD:refs/heads/development) "
                    "from branch development is allowed in Cursor direct mode."
                ),
            }
        reason = validate_development_push(_repo_root(cmd))
        if reason:
            return {"permission": "deny", "user_message": reason}
    return None


def decide(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return {"permission": "deny", "user_message": "Cursor integration hook received invalid shell payload."}
    cmd = _normalize(value["command"])
    push_block = _push_denial(cmd)
    if push_block:
        return push_block
    if any(pattern.search(cmd) for pattern in BANNED) or _gh_api_implicit_post(cmd):
        return {
            "permission": "deny",
            "user_message": "Command blocked by Cursor shell guard (destructive git or direct gh mutation).",
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
