# © Artur Czarnecki. All rights reserved.

"""EBH-7 W10 Protocol 03 package gates (B6 operator/independent)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]


def test_ebh7_w10_protocol_03_package_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_PROTOCOL_03_EXECUTION_PACKAGE.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for step in ("B1", "B2", "B3", "B4", "B5", "B6"):
        assert step in text
    assert "B6" in text and "OPEN" in text
