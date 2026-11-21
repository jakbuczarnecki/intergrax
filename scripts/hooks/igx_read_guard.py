# © Artur Czarnecki. All rights reserved.
"""PreToolUse guard for ``Read``: deny un-ranged reads of large files and repeated identical reads.

Usage (settings.json hooks)::

    python scripts/hooks/igx_read_guard.py            # PreToolUse on Read
    python scripts/hooks/igx_read_guard.py epoch      # SessionStart(compact|resume), PostCompact

Context epoch: advances on SessionStart(compact|resume) and PostCompact; a repeat is only a repeat
within one epoch. An identical retry immediately after a denial is allowed (insist). The ``auditor``
role (``IGX_ROLE=auditor``) is exempt from the repeat rule. Files under ``tool-results/`` always
count as large. ``IGX_HOOKS=off`` disables the guard.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import igx_hooklib as lib  # noqa: E402

_HEADING = re.compile(r"^(#{1,6}\s|\s*(?:async\s+def|def|class)\s)")
_SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ipynb"}
MAX_OUTLINE = 30


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


def _is_large(path: Path, size: int) -> bool:
    if "tool-results" in path.parts:
        return True
    return size > lib.MAX_BYTES or lib.count_lines(path) > lib.MAX_LINES


def main(argv: list[str]) -> int:
    if lib.disabled():
        return 0
    data = lib.load_input()
    session = str(data.get("session_id", ""))
    state = lib.load_state(session)

    if argv[1:2] == ["epoch"]:
        state["epoch"] = int(state.get("epoch", 0)) + 1
        state["seen"], state["last_denied"] = {}, None
        lib.save_state(session, state)
        return 0

    if data.get("tool_name") != "Read":
        return 0
    args = data.get("tool_input") or {}
    raw = args.get("file_path")
    if not isinstance(raw, str) or not raw:
        return 0
    path = Path(raw)
    if path.suffix.lower() in _SKIP_SUFFIX:
        return 0
    try:
        st = path.stat()
    except OSError:
        return 0
    offset, limit = args.get("offset"), args.get("limit")
    ranged = offset is not None or limit is not None
    key = f"{path}|{offset}|{limit}"
    fingerprint = [st.st_mtime_ns, st.st_size, int(state.get("epoch", 0))]

    reason = ""
    if not ranged and _is_large(path, st.st_size):
        reason = (
            f"{path.name} is large ({st.st_size} bytes). Read a range with offset/limit or Grep first."
            f" Outline: {_outline(path) or 'n/a'}"
        )
    elif (state.get("seen") or {}).get(key) == fingerprint and os.environ.get("IGX_ROLE") != "auditor":
        reason = f"Repeat read of {path.name} (same range, file unchanged since the last read in this context)."

    if reason:
        if state.get("last_denied") == key:  # insist: identical retry right after a denial
            state["last_denied"] = None
        else:
            state["last_denied"] = key
            lib.save_state(session, state)
            return lib.deny(reason)
    else:
        state["last_denied"] = None
    state.setdefault("seen", {})[key] = fingerprint
    lib.save_state(session, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
