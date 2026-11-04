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
