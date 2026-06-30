#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Release eval gates for agent roster promotion (architecture §40.9 · ACP-PROD-9)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
_CI_DIR = REPO_ROOT / "scripts" / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))
from script_paths import resolve_script  # noqa: E402


def _run(script: str, *args: str) -> int:
    path = resolve_script(script)
    for cmd in (
        ["uv", "run", "python", str(path), *args],
        [PYTHON, str(path), *args],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    return 1


def main() -> int:
    violations: list[str] = []

    if _run("check_agent_acp_close_ci.py") != 0:
        violations.append("check_agent_acp_close_ci.py failed")

    if _run("check_agent_threat_model.py") != 0:
        violations.append("check_agent_threat_model.py failed")

    if _run("check_acp_ci_conformance_matrix.py", "--scripts-only") != 0:
        violations.append("check_acp_ci_conformance_matrix.py failed")

    if violations:
        print("Agent release gate violations:")
        print("\n".join(violations))
        return 1

    print("Agent release gates: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
