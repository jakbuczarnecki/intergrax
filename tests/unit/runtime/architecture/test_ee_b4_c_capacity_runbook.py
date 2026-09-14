# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — capacity runbook directs diagnosis not permit manipulation."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.operations.runbook_contract import RunbookSectionId
from testing_support.operations.runbook_validator import (
    get_runbook_body,
    runbook_section_text,
)

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


def test_ee_b4_c_capacity_runbook_uses_capacity_readiness_diagnosis() -> None:
    body = get_runbook_body(_RUNBOOKS, "RB-02")
    diagnosis = runbook_section_text(body, RunbookSectionId.DIAGNOSIS).lower()
    immediate = runbook_section_text(
        body, RunbookSectionId.IMMEDIATE_SAFE_ACTION
    ).lower()
    assert "capacity" in diagnosis or "capacity" in immediate
    assert "assess_root_execution_capacity" in body or "capacity/readiness" in immediate
    assert "clear internal permit" not in immediate
    assert "manually increment capacity" not in immediate
