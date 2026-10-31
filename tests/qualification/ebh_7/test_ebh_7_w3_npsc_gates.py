# © Artur Czarnecki. All rights reserved.

"""EBH-7 W3 NPSC remediation gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]


def test_ebh7_w3_npsc_record_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_NPSC_ALLOWLIST_REFRESH_RECORD.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "DUP-X B4" in text


def test_ebh7_w3_npsc_global_gate_passes() -> None:
    from tests.unit.runtime.architecture.test_npsc4_2_residual_compatibility_gate import (
        test_npsc42_orchestration_backend_access_confined_to_allowlist,
    )

    test_npsc42_orchestration_backend_access_confined_to_allowlist()
