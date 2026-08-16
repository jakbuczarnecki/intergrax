# © Artur Czarnecki. All rights reserved.

"""BUILDER-CONVERSION-P0-2: executable Builder Quick Start public contracts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
QUICKSTART_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILDER_QUICKSTART.md"
README_PATH = REPO_ROOT / "README.md"
SCAFFOLD_CLI = REPO_ROOT / "intergrax" / "scaffold" / "cli.py"

_PRIMARY_SCAFFOLD_CMD = "python -m intergrax.scaffold new-stack"
_FORBIDDEN_POSITIVE_CLAIMS = (
    "production-ready generated",
    "stable public sdk",
    "5-minute onboarding",
    "zero-config setup",
    "finished developer platform",
)


@pytest.fixture(scope="module")
def quickstart_text() -> str:
    return QUICKSTART_PATH.read_text(encoding="utf-8")


def test_builder_quickstart_contains_primary_scaffold_command(quickstart_text: str) -> None:
    assert _PRIMARY_SCAFFOLD_CMD in quickstart_text
    assert "--profile lab" in quickstart_text
    assert quickstart_text.count("new-stack") >= 1
    assert "new-application" in quickstart_text
    assert "new-agent" in quickstart_text


def test_builder_quickstart_documents_verification_commands(quickstart_text: str) -> None:
    assert "uv run pytest applications/my_first_stack_application/tests -q" in quickstart_text
    assert "uv run uvicorn my_first_stack_application.host.main:app" in quickstart_text
    assert "POST /run" in quickstart_text
    assert "/debug/tasks/" in quickstart_text


def test_builder_quickstart_single_primary_route(quickstart_text: str) -> None:
    """First scaffold section should not present three equivalent top-level starting paths."""
    scaffold_section = quickstart_text.split("## Scaffold the stack", 1)[1].split("## What was generated", 1)[0]
    assert scaffold_section.count("python -m intergrax.scaffold new-stack") == 1
    assert "Prefer `new-stack`" in quickstart_text


def test_builder_quickstart_linked_files_resolve(quickstart_text: str) -> None:
    base = QUICKSTART_PATH.parent
    link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    for _label, target in link_pattern.findall(quickstart_text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            continue
        clean = target.split("#", 1)[0].strip()
        if not clean:
            continue
        resolved = (base / clean).resolve()
        assert resolved.exists(), f"Broken link target: {target}"


def test_readme_start_building_routes_to_builder_quickstart() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    assert "AI Engineer / Builder" in readme
    assert "docs/project/builders/BUILDER_QUICKSTART.md" in readme


def test_builder_quickstart_does_not_claim_production_readiness(quickstart_text: str) -> None:
    lower = quickstart_text.lower()
    assert "not production-ready" in lower or "not a production" in lower
    negation_markers = ("not ", "no ", "does not", "do not", "without")
    for phrase in _FORBIDDEN_POSITIVE_CLAIMS:
        idx = lower.find(phrase)
        if idx == -1:
            continue
        context = lower[max(0, idx - 50) : idx + len(phrase) + 30]
        assert any(marker in context for marker in negation_markers), (
            f"Positive forbidden claim {phrase!r} at index {idx}"
        )


def test_scaffold_new_stack_command_registered_in_cli() -> None:
    text = SCAFFOLD_CLI.read_text(encoding="utf-8")
    assert '"new-stack"' in text
    assert "run_new_stack" in text


def test_documented_scaffold_command_exits_zero_in_temp_repo(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "applications").mkdir(parents=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "intergrax.scaffold",
            "new-stack",
            "builder_qs_contract",
            "--profile",
            "lab",
            "--root",
            str(root),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (root / "agents" / "builder_qs_contract").is_dir()
    assert (root / "applications" / "builder_qs_contract_application" / "manifest.py").is_file()
