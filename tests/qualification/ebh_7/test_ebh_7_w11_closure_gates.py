# © Artur Czarnecki. All rights reserved.

"""EBH-7 W11 closure handoff gates (OD-EBH7-08 remains operator-owned)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]


def test_ebh7_w11_evol_handoff_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_EVOL_X_HANDOFF_RECORD.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "PROD-Q" in text
    assert "global FRZ" in text.lower() or "FRZ PASS" in text


def test_ebh7_w11_stage_readiness_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_STAGE_CERTIFICATION_READINESS.md"
    assert path.is_file()


def test_ebh7_w11_od_08_not_self_approved() -> None:
    text = (_REPO / "docs/project/maintainers/qualification/EBH_7_OPERATOR_DECISION_REGISTER.md").read_text(
        encoding="utf-8"
    )
    assert "OD-EBH7-08" in text
    assert "OPEN" in text
