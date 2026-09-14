# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — mandatory evidence fail closed vs observability degraded."""

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


def test_ee_b4_c_mandatory_evidence_fail_closed() -> None:
    body = get_runbook_body(_RUNBOOKS, "RB-03")
    assert "FAIL CLOSED" in body


def test_ee_b4_c_observability_export_degraded_not_stop_engine() -> None:
    body = get_runbook_body(_RUNBOOKS, "RB-04")
    assert "DEGRADED" in body
    assert "do not stop" in body.lower() or "do not stop execution" in body.lower()
