#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Validate Tier-0 catalog bootstrap (Phase P-Ext.4)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _python_for_pytest(repo_root: Path) -> list[str]:
    scripts = repo_root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    for name in ("python.exe", "python") if sys.platform == "win32" else ("python",):
        candidate = scripts / name
        if candidate.is_file():
            return [str(candidate), "-m", "pytest"]
    return [sys.executable, "-m", "pytest"]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tests = [
        "tests/unit/core/plugins/test_catalog_bootstrap.py",
        "tests/unit/core/plugins/test_catalog_conflict_policy.py",
        "tests/unit/core/plugins/test_entry_point_catalog_bootstrap.py",
        "tests/unit/core/plugins/test_plugin_catalog_counts.py",
        "tests/unit/integrations/test_external_plugin.py",
        "tests/unit/integrations/test_external_integration_entry_point.py",
        "tests/unit/integrations/test_resolve_typed.py",
        "tests/unit/tools/test_external_tool_plugin.py",
        "tests/unit/skills/test_external_skill_plugin.py",
    ]
    cmd = [*_python_for_pytest(repo_root), *tests, "-q", "--tb=short"]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    if result.returncode != 0:
        print("ERROR: plugin catalog smoke tests failed", file=sys.stderr)
        return 1
    print("OK plugin catalog smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
