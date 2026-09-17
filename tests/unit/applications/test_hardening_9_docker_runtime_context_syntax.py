# © Artur Czarnecki. All rights reserved.

"""HARDENING-9 — Docker runtime-context must not block canonical application syntax gates."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from intergrax.applications._shared.application_build_context import (
    ApplicationPythonSyntaxError,
    validate_canonical_application_python_syntax,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


@pytest.mark.gate
def test_canonical_application_python_syntax_valid_repo() -> None:
    """Canonical Tier-3 application sources in the real repo must parse."""
    validate_canonical_application_python_syntax(REPO)


@pytest.mark.gate
def test_stale_runtime_context_ignored_by_canonical_gate(tmp_path: Path) -> None:
    """Materialized ``docker/runtime-context`` on disk is not canonical source."""
    root = tmp_path / "repo"
    _write(root / "applications" / "foo" / "host.py", "OK = True\n")
    _write(
        root / "applications" / "foo" / "docker" / "runtime-context" / "broken.py",
        "def broken(\n",
    )
    validate_canonical_application_python_syntax(root)


@pytest.mark.gate
def test_invalid_canonical_source_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "applications" / "example" / "broken.py", "def broken(\n")
    with pytest.raises(
        ApplicationPythonSyntaxError, match="APPLICATION_CANONICAL_PYTHON_SYNTAX_FAILED"
    ):
        validate_canonical_application_python_syntax(root)
