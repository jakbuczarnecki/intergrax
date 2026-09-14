# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — compound failure closure (delegates to EE-B2 compound scenarios)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_COMPOUND = (
    _REPO
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_ee_b2_compound_failure.py"
)


def test_ee_b2_final_compound_module_present() -> None:
    assert _COMPOUND.is_file()


def test_ee_b2_final_compound_module_covers_two_scenarios() -> None:
    text = _COMPOUND.read_text(encoding="utf-8")
    assert "test_ee_b2_compound_worker_failure_plus_otlp_export_failure" in text
    assert (
        "test_ee_b2_compound_execution_failure_plus_mandatory_evidence_failure" in text
    )
    assert "test_ee_b2_compound_capacity_saturation_plus_cancel_releases_slot" in text
