# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACP = ROOT / "governance" / "agent-control-plane"
GOV = ROOT / "scripts" / "governance"


def test_validate_policy_registry_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOV / "validate_policy_registry.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_validate_policy_registry_detects_missing_artifact(tmp_path: Path) -> None:
    reg = tmp_path / "policy-registry.yaml"
    reg.write_text(
        "policies:\n  bad:\n    id: X\n    artifacts:\n      - does/not/exist.md\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(ROOT))
    from scripts.governance.validate_policy_registry import validate_registry

    errors = validate_registry(repo=tmp_path, registry_path=reg)
    assert any("missing artifact" in e for e in errors)


def test_verify_trusted_policy_pin_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_trusted_policy_pin.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_trusted_policy_pin_uses_git_blob_not_worktree_newlines() -> None:
    """Pins must match HEAD blob bytes (CI/Linux LF), not CRLF working-tree on Windows."""
    import hashlib
    import subprocess

    rel = "governance/GOVERNANCE-MANIFEST.yaml"
    wt = (ROOT / rel).read_bytes()
    blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{rel}"], cwd=ROOT).strip()
    head = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
    if wt == head:
        return
    assert hashlib.sha256(wt).hexdigest() != hashlib.sha256(head).hexdigest()
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_trusted_policy_pin.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0
