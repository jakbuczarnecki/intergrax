#!/usr/bin/env python3
"""Antigravity PreToolUse safety guard for Intergrax (stdlib only).

Mirrors the Intergrax safety invariants enforced in Claude Code / Cursor without
modifying either environment:
- read_file: deny secret .env files (.env, .env.local, .env.production); deny
  un-ranged reads of very large text files (>80 KB or >1500 lines) with an outline.
- run_command: deny destructive git / governance bypasses (force push, reset --hard,
  --no-verify, INTERGRAX_SKIP_*) and whole-file dumps of >80 KB files; prompt (`ask`)
  on branch-switching commands and human-boundary remote mutations (git push, gh pr).
- invoke_subagent: deny generic general-purpose aliases; prompt (`ask`) on `self`
  so typed igx-* roles are preferred.

Kill switch: set `IGX_HOOKS=off` to allow all tool calls.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

MAX_BYTES = 80 * 1024
MAX_LINES = 1500
MAX_OUTLINE = 30

_DENIED_ENV_NAMES = frozenset({".env", ".env.local", ".env.production"})
_SKIP_SUFFIX = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ipynb", ".mp4", ".mp3", ".wav"})
_HEADING = re.compile(r"^(#{1,6}\s|\s*(?:async\s+def|def|class)\s)")

_DUMP_CMDS = frozenset({"cat", "type", "get-content", "gc"})
_SPLIT = re.compile(r"&&|\|\||;|\||\r?\n")

_HARD_DENY_CMD = (
    (
        re.compile(
            r"(?i)\bgit(?:\.exe)?\b[^\n;&|]*\bpush\b[^\n;&|]*(?:\s--force(?:-with-lease)?\b|\s-[A-Za-z]*f\b|\s\+[^\s]+)"
        ),
        "Force push is forbidden in Intergrax (role_registry.yaml global_denied_commands).",
    ),
    (
        re.compile(r"(?i)\bgit(?:\.exe)?\b[^\n;&|]*\breset\b[^\n;&|]*\s--hard\b"),
        "`git reset --hard` is forbidden in Intergrax (role_registry.yaml global_denied_commands).",
    ),
    (re.compile(r"(?i)\b--no-verify\b"), "`--no-verify` hook bypass is forbidden in Intergrax."),
    (
        re.compile(r"\bINTERGRAX_SKIP_[A-Z0-9_]*"),
        "`INTERGRAX_SKIP_*` governance bypass variables are forbidden in Intergrax.",
    ),
)

_ASK_CMD = (
    (
        re.compile(r"(?i)\bgit(?:\.exe)?\b[^\n;&|]*\bcheckout\b[^\n;&|]*(?:\s-b\b|\s--detach\b)"),
        "Branch creation/detach on the main checkout requires operator approval (see GEMINI.md / igx-worktree).",
    ),
    (
        re.compile(r"(?i)\bgit(?:\.exe)?\b[^\n;&|]*\bswitch\b[^\n;&|]*(?:\s-c\b|\s--detach\b)"),
        "Branch creation/detach on the main checkout requires operator approval (see GEMINI.md / igx-worktree).",
    ),
    (
        re.compile(r"(?i)\bgit(?:\.exe)?\b[^\n;&|]*\bpush\b"),
        "`git push` is a human boundary in Intergrax unless explicitly authorized.",
    ),
    (
        re.compile(r"(?i)\bgh(?:\.exe)?\b[^\n;&|]*\bpr\b[^\n;&|]*\b(?:create|new|merge)\b"),
        "GitHub PR creation/merge is a human boundary in Intergrax unless explicitly authorized.",
    ),
)

_DENIED_SUBAGENTS = frozenset({"general-purpose", "generalPurpose", "GeneralPurpose"})
_TYPED_ROLES = "igx-researcher, igx-architect, igx-implementer, igx-tester, igx-reviewer"


def _emit(decision: str, reason: str = "") -> int:
    payload: dict[str, Any] = {"decision": decision}
    if reason:
        payload["reason"] = reason
    sys.stdout.write(json.dumps(payload))
    return 0


def _count_lines(path: Path, limit: int = MAX_LINES + 1) -> int:
    count = 0
    try:
        with path.open("rb") as fh:
            for _ in fh:
                count += 1
                if count >= limit:
                    break
    except OSError:
        return 0
    return count


def _outline(path: Path) -> str:
    found: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for no, line in enumerate(fh, 1):
                if _HEADING.match(line):
                    found.append(f"{no}: {line.strip()[:60]}")
                    if len(found) >= MAX_OUTLINE:
                        break
    except OSError:
        return ""
    return "; ".join(found)


def _check_read_file(args: dict[str, Any], cwd: str) -> tuple[str, str]:
    raw = args.get("Path") or args.get("file_path") or ""
    if not isinstance(raw, str) or not raw.strip():
        return "allow", ""
    path = Path(raw)
    if not path.is_absolute() and cwd:
        path = Path(cwd) / path
    if path.name.lower() in _DENIED_ENV_NAMES:
        return "deny", f"Reading secret file `{path.name}` is denied by Intergrax workspace policy."
    if path.suffix.lower() in _SKIP_SUFFIX:
        return "allow", ""
    ranged = args.get("StartLine") is not None or args.get("EndLine") is not None
    if ranged:
        return "allow", ""
    try:
        st = path.stat()
    except OSError:
        return "allow", ""
    if st.st_size > MAX_BYTES or _count_lines(path) > MAX_LINES:
        outline = _outline(path) or "n/a"
        return (
            "deny",
            f"{path.name} is large ({st.st_size} bytes). Pass StartLine/EndLine or search first. Outline: {outline}",
        )
    return "allow", ""


def _check_large_dump(command: str, cwd: str) -> str:
    for part in _SPLIT.split(command):
        piece = part.strip()
        if not piece:
            continue
        try:
            tokens = shlex.split(piece, posix=False)
        except ValueError:
            continue
        if not tokens:
            continue
        tokens = [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
        if tokens[0].lower() not in _DUMP_CMDS:
            continue
        for arg in tokens[1:]:
            if arg.startswith("-"):
                continue
            target = Path(arg)
            if not target.is_absolute():
                target = Path(cwd or ".") / target
            try:
                size = target.stat().st_size
            except OSError:
                continue
            if size > MAX_BYTES:
                return (
                    f"{tokens[0]} of {target.name} ({size} bytes) is too large. Use read_file with StartLine/EndLine."
                )
    return ""


def _check_run_command(args: dict[str, Any], cwd: str) -> tuple[str, str]:
    cmd = str(args.get("CommandLine") or args.get("command") or "")
    if not cmd.strip():
        return "allow", ""
    for pattern, reason in _HARD_DENY_CMD:
        if pattern.search(cmd):
            return "deny", reason
    dump_reason = _check_large_dump(cmd, cwd)
    if dump_reason:
        return "deny", dump_reason
    for pattern, reason in _ASK_CMD:
        if pattern.search(cmd):
            return "ask", reason
    return "allow", ""


def _check_invoke_subagent(args: dict[str, Any]) -> tuple[str, str]:
    sub_type = str(args.get("TypeName") or "").strip()
    if sub_type in _DENIED_SUBAGENTS:
        return "deny", f"Generic subagent `{sub_type}` is denied. Use a typed Intergrax role: {_TYPED_ROLES}."
    if sub_type == "self":
        return (
            "ask",
            f"Delegating to generic `self` subagent. Intergrax prefers typed roles ({_TYPED_ROLES}); "
            "approve to proceed.",
        )
    return "allow", ""


def decide(data: dict[str, Any]) -> tuple[str, str]:
    tool_call = data.get("toolCall")
    if not isinstance(tool_call, dict):
        return "allow", ""
    name = str(tool_call.get("name") or "")
    args = tool_call.get("args")
    if not isinstance(args, dict):
        args = {}
    cwd = str(data.get("cwd") or "")
    if name == "read_file":
        return _check_read_file(args, cwd)
    if name == "run_command":
        return _check_run_command(args, cwd)
    if name == "invoke_subagent":
        return _check_invoke_subagent(args)
    return "allow", ""


def main() -> int:
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return _emit("allow")
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            return _emit("allow")
        decision, reason = decide(data)
        return _emit(decision, reason)
    except Exception:  # noqa: BLE001 - fail open on unexpected parser error
        return _emit("allow")


if __name__ == "__main__":
    raise SystemExit(main())
