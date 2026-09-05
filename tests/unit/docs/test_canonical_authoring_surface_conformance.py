# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.maintenance.check_canonical_authoring_surface_conformance import (
    REPO_ROOT,
    audit_repository,
    main,
    scan_surface,
)

pytestmark = pytest.mark.gate

CHECKER = REPO_ROOT / "scripts" / "maintenance" / "check_canonical_authoring_surface_conformance.py"
GUIDE = REPO_ROOT / "docs" / "project" / "technical" / "guides" / "AGENT_CREATION_GUIDE.md"


def test_canonical_surfaces_pass_current_repo() -> None:
    assert audit_repository() == []


def test_gate_detects_registry_register_in_guide_line() -> None:
    sample = "registry = AgentRegistry()\nregistry.register(MyAgent())\n"
    violations = scan_surface_from_text(sample)
    assert any("registry.register(" in item for item in violations)


def test_gate_allows_do_not_instruction_line() -> None:
    sample = "Do not call registry.register() on serving paths.\n"
    assert scan_surface_from_text(sample) == []


def test_main_passes_for_current_repo() -> None:
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Canonical authoring surface conformance: OK" in proc.stdout
    assert main([]) == 0


def scan_surface_from_text(text: str) -> list[str]:
    tmp = REPO_ROOT / ".tmp" / "session" / "stage16" / "gate_probe.md"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    try:
        return scan_surface(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
