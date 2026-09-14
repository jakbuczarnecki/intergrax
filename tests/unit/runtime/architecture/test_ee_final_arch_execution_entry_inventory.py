# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — execution entry inventory and convergence documentation."""

from __future__ import annotations

import re

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import (
    ARCH_MODEL,
    ARCH_MODEL_REQUIRED_SECTIONS,
    EE_FINAL_ARCH_GATE_MODULES,
    P0_INVENTORY,
    QUALIFICATION,
    _REPO_ROOT,
)
from tests.unit.runtime.architecture.test_platform_execution_unification_p0_bypass_inventory import (
    _EXPECTED_ENTRYPOINT_COUNT,
    _inventory_doc_text,
    _parse_central_inventory_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_final_arch_model_and_qualification_present() -> None:
    assert ARCH_MODEL.is_file()
    assert QUALIFICATION.is_file()
    text = ARCH_MODEL.read_text(encoding="utf-8")
    for heading in ARCH_MODEL_REQUIRED_SECTIONS:
        assert heading in text, f"missing {heading}"


def test_ee_final_arch_p0_inventory_entrypoint_count_matches_frozen_baseline() -> None:
    rows = _parse_central_inventory_rows(_inventory_doc_text())
    assert len(rows) == _EXPECTED_ENTRYPOINT_COUNT


def test_ee_final_arch_all_gate_modules_present() -> None:
    missing = [
        rel for rel in EE_FINAL_ARCH_GATE_MODULES if not (_REPO_ROOT / rel).is_file()
    ]
    assert missing == []


def test_ee_final_arch_qualification_references_p0_inventory_ssot() -> None:
    qual = QUALIFICATION.read_text(encoding="utf-8")
    assert "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md" in qual
    p0 = P0_INVENTORY.read_text(encoding="utf-8")
    supported = re.search(
        r"^\| Supported execution bypasses \(production\) \| (\d+) \|",
        p0,
        flags=re.MULTILINE,
    )
    assert supported is not None
    assert int(supported.group(1)) == 0
