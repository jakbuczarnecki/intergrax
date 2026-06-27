#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""UAEP-AUDIT-01 — RuntimeEvent emitters must populate tenant_id (§42.44.2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EMITTER_FILES = (
    REPO_ROOT / "intergrax/agents/uaep.py",
    REPO_ROOT / "intergrax/runtime/middleware/trace_middleware.py",
)

# RuntimeEvent( ... ) must include tenant_id= on hot-path emitters.
EVENT_CONSTRUCTOR = re.compile(r"RuntimeEvent\s*\(")
TENANT_KWARG = re.compile(r"tenant_id\s*=")


def _check_file(path: Path) -> list[str]:
    if not path.is_file():
        return [f"missing emitter file: {path.relative_to(REPO_ROOT).as_posix()}"]
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []
    for match in EVENT_CONSTRUCTOR.finditer(text):
        start = match.start()
        line_no = text.count("\n", 0, start) + 1
        snippet = text[start : start + 600]
        if not TENANT_KWARG.search(snippet):
            violations.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}:{line_no}: "
                "RuntimeEvent missing tenant_id="
            )
    return violations


def main() -> int:
    violations: list[str] = []
    for path in EMITTER_FILES:
        violations.extend(_check_file(path))
    if violations:
        print("runtime event tenant propagation audit failed:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("runtime event tenant propagation audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
