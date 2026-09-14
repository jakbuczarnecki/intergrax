# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — unknown side effect requires no blind retry."""

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


def test_ee_b4_c_unknown_side_effect_no_blind_retry() -> None:
    body = get_runbook_body(_RUNBOOKS, "RB-11")
    assert "NO BLIND RETRY" in body
