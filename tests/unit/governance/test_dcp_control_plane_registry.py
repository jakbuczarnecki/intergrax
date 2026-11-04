# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
ACP = ROOT / "governance" / "agent-control-plane"
GOV = ROOT / "scripts" / "governance"


def test_role_registry_seven_roles() -> None:
    doc = yaml.safe_load((ACP / "role_registry.yaml").read_text(encoding="utf-8"))
    assert doc["default_role"] == "orchestrator"
    assert len(doc["roles"]) == 7


def test_verify_agent_control_plane_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_agent_control_plane.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_verify_control_plane_consistency_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_control_plane_consistency.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_verify_agent_skills_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_agent_skills.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_plan_digest_stale_fails_verification(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\ncontent v1\n", encoding="utf-8")
    ticket = tmp_path / "ticket.md"
    ticket.write_text("Plan digest: " + "0" * 64 + "\nApproved: op @ 2026-01-01\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(GOV / "verify_plan_approval.py"),
            str(ticket),
            "--plan",
            str(plan),
        ],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 1
