# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — forbidden operator bypass patterns in action sections."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.operations.runbook_validator import validate_runbook_document

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_RUNBOOKS = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "runbooks"
    / "EXECUTION_ENGINE_PRODUCTION_RUNBOOKS.md"
)


def test_ee_b4_c_no_forbidden_bypass_in_operator_action_sections() -> None:
    results = validate_runbook_document(_RUNBOOKS)
    offenders = [
        f"{r.runbook_id}:{','.join(r.forbidden_hits)}"
        for r in results
        if r.forbidden_hits
    ]
    assert offenders == []
