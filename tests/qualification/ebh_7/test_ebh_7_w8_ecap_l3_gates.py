# © Artur Czarnecki. All rights reserved.

"""EBH-7 W8 ECAP C1–C5 + consolidated L3 gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]


def test_ebh7_w8_ecap_report_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for label in ("ECAP-C1", "ECAP-C2", "ECAP-C3", "ECAP-C4", "ECAP-C5"):
        assert label in text


def test_ebh7_w8_l3_consolidated_suite_importable() -> None:
    import tests.qualification.ebh_6.test_ebh_6_mechanical_exit_gates as ebh6_exit

    assert hasattr(ebh6_exit, "test_ebh6_exit_findings_register_complete")
