# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-6: forbid legacy direct model-context injection in canonical Nexus runtime."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = (
    REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context",
    REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools",
    REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine",
    REPO_ROOT / "intergrax" / "runtime" / "execution",
)

FORBIDDEN_CALLS = (
    "insert_context_before_last_user",
    "append_native_tool_messages",
    "inject_tool_traces_system_context",
    "search_user_longterm_memory",
)

SKIP_FILES = frozenset(
    {
        "tool_context_helpers.py",
        "tool_loop.py",
        "session_manager.py",
    }
)


def _count_calls(path: Path, symbol: str) -> int:
    text = path.read_text(encoding="utf-8")
    return len(re.findall(rf"\b{re.escape(symbol)}\s*\(", text))


def main() -> int:
    violations: list[str] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.name in SKIP_FILES:
                continue
            for symbol in FORBIDDEN_CALLS:
                count = _count_calls(path, symbol)
                if count:
                    violations.append(f"{path.relative_to(REPO_ROOT)}: {symbol} calls={count}")
    if violations:
        sys.stderr.write("MEM-XINT-6 canonical bypass guard failed:\n")
        sys.stderr.write("\n".join(violations) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
