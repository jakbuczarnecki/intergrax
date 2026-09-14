# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — recovery runbook forbids manual checkpoint mutation."""

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


def test_ee_b4_c_recovery_runbook_no_manual_checkpoint_edit_in_recovery() -> None:
    body = get_runbook_body(_RUNBOOKS, "RB-07")
    recovery = runbook_section_text(body, RunbookSectionId.RECOVERY_PATH).lower()
    diagnosis = runbook_section_text(body, RunbookSectionId.DIAGNOSIS).lower()
    combined = recovery + diagnosis
    assert "edit checkpoint" not in combined
    assert "manual mutation" not in combined or "no manual mutation" in body.lower()
