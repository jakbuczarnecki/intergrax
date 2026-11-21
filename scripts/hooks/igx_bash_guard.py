# © Artur Czarnecki. All rights reserved.
"""PreToolUse guard for ``Bash``: deny whole-file dumps of large files and wide ``sed -n`` ranges.

Denies only ``cat``/``type``/``Get-Content`` on files over 80 KB and ``sed -n 'A,Bp'`` ranges over
600 lines. ``IGX_HOOKS=off`` disables the guard.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import igx_hooklib as lib  # noqa: E402

MAX_SED_RANGE = 600
_DUMP = {"cat", "type", "get-content", "gc"}
_RANGE = re.compile(r"^(\d+),(\d+)p$")
_SPLIT = re.compile(r"&&|\|\||;|\|")


def _segments(command: str) -> list[list[str]]:
    out: list[list[str]] = []
    for part in _SPLIT.split(command):
        try:
            tokens = shlex.split(part.strip(), posix=True)
        except ValueError:
            continue
        if tokens:
            out.append(tokens)
    return out


def _resolve(raw: str, cwd: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else Path(cwd or ".") / path


def check(command: str, cwd: str) -> str:
    for tokens in _segments(command):
        name = tokens[0].lower()
        if name in _DUMP:
            for arg in tokens[1:]:
                if arg.startswith("-"):
                    continue
                try:
                    size = _resolve(arg, cwd).stat().st_size
                except OSError:
                    continue
                if size > lib.MAX_BYTES:
                    return f"{name} of {Path(arg).name} ({size} bytes) is too large. Use Read with offset/limit."
        elif name == "sed" and "-n" in tokens:
            for arg in tokens[1:]:
                m = _RANGE.match(arg)
                if m and int(m.group(2)) - int(m.group(1)) + 1 > MAX_SED_RANGE:
                    return f"sed -n range {arg} spans over {MAX_SED_RANGE} lines. Use Read with offset/limit."
    return ""


def main() -> int:
    if lib.disabled():
        return 0
    data = lib.load_input()
    if data.get("tool_name") != "Bash":
        return 0
    command = str((data.get("tool_input") or {}).get("command", ""))
    reason = check(command, str(data.get("cwd", "")))
    return lib.deny(reason) if reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
