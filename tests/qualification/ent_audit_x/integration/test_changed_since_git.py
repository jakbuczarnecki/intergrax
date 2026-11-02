from __future__ import annotations

import subprocess
from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import select_checks_for_changed_since
from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_changed_since_uses_git_diff() -> None:
    root = Path(__file__).resolve().parents[4]
    parent = subprocess.check_output(["git", "rev-parse", "HEAD~1"], cwd=root, text=True).strip()
    checks = load_default_manifest()
    selected = select_checks_for_changed_since(checks, repo_root=root, since_sha=parent)
    assert isinstance(selected, frozenset)
    assert len(selected) >= len({c.check_id for c in checks if c.check_id.startswith("TEN-")})
