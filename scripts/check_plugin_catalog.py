#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Validate Tier-0 catalog bootstrap (Phase P-Ext.4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tests = [
        "tests/unit/core/plugins/test_catalog_bootstrap.py",
        "tests/unit/core/plugins/test_plugin_catalog_counts.py",
        "tests/unit/integrations/test_external_plugin.py",
        "tests/unit/tools/test_external_tool_plugin.py",
        "tests/unit/skills/test_external_skill_plugin.py",
    ]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-q",
        "--tb=short",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(repo_root)},
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: plugin catalog smoke tests failed", file=sys.stderr)
        return 1
    print("OK plugin catalog smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
