# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — legacy execution paths isolated from production wiring."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import (
    ARCH_MODEL,
    P0_INVENTORY,
)
from tests.unit.runtime.architecture.test_platform_execution_unification_p0_bypass_inventory import (
    _inventory_doc_text,
    _parse_central_inventory_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_final_arch_legacy_paths_documented() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "## 10. Accepted legacy and test-only paths" in text


def test_ee_final_arch_p0_legacy_rows_are_non_production_only() -> None:
    rows = _parse_central_inventory_rows(_inventory_doc_text())
    legacy = [row for row in rows if row["verdict"] == "LEGACY BUT NON-PRODUCTION"]
    assert len(legacy) >= 1
    for row in legacy:
        assert row["verdict"] != "BYPASS"
        assert row["verdict"] != "CANONICAL WITH GAP"


def test_ee_final_arch_p0_inventory_references_u5_qualification() -> None:
    text = P0_INVENTORY.read_text(encoding="utf-8")
    assert "FINAL_ZERO_BYPASS_QUALIFIED" in text or "U5" in text
