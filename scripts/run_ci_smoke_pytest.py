#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Run CI smoke unit tests without collecting the full tests/unit tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.unit.conftest import CI_SMOKE_DIR_PREFIXES, CI_SMOKE_FILES  # noqa: E402


def smoke_paths() -> list[str]:
    return [*CI_SMOKE_DIR_PREFIXES, *sorted(CI_SMOKE_FILES)]


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *smoke_paths(),
        "-m",
        "ci_smoke",
        "-q",
        "--tb=line",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=_REPO)


if __name__ == "__main__":
    raise SystemExit(main())
