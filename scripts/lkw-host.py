#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW hybrid daemon launcher (CFG-14 / AUDIT-IDEAL-28.3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "applications", ROOT / "agents"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from local_workspace_application.host.main import run  # noqa: E402


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
