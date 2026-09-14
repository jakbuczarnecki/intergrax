# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — runbook presence and required sections."""

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


def test_ee_b4_c_all_runbooks_present_with_sections() -> None:
    results = validate_runbook_document(_RUNBOOKS)
    assert len(results) == 14
    for result in results:
        assert result.missing_sections == (), (
            f"{result.runbook_id} missing {result.missing_sections}"
        )
