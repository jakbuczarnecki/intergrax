# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — shutdown runbook reuses EE-B4-B contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.operations.runbook_validator import get_runbook_body

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_RUNBOOKS = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "runbooks"
    / "EXECUTION_ENGINE_PRODUCTION_RUNBOOKS.md"
).read_text(encoding="utf-8")
_SHUTDOWN_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_GRACEFUL_SHUTDOWN_DRAIN_TERMINATION_MODEL.md"
)


def test_ee_b4_c_shutdown_runbook_references_ee_b4_b() -> None:
    body = get_runbook_body(_RUNBOOKS, "RB-13")
    assert "EE-B4-B" in body
    assert "ExecutionRuntimeShutdownPhase" in body
    assert _SHUTDOWN_MODEL.is_file()
