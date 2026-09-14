# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — zero supported production execution bypass."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import (
    p0_bypass_count,
    scan_forbidden_imports_in_production,
)
from tests.unit.runtime.architecture.test_platform_execution_unification_p0_bypass_inventory import (
    _central_inventory_verdict_counts,
    _inventory_doc_text,
    _parse_central_inventory_rows,
    _parse_metrics_verdict_counts,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_final_arch_p0_inventory_bypass_is_zero() -> None:
    assert p0_bypass_count() == 0


def test_ee_final_arch_inventory_metrics_align_with_rows() -> None:
    text = _inventory_doc_text()
    rows = _parse_central_inventory_rows(text)
    inventory_counts = _central_inventory_verdict_counts(rows)
    metrics_counts = _parse_metrics_verdict_counts(text)
    for label, count in inventory_counts.items():
        assert metrics_counts[label] == count
    assert inventory_counts["BYPASS"] == 0
    assert inventory_counts["AMBIGUOUS"] == 0


def test_ee_final_arch_production_trees_avoid_forbidden_execution_imports() -> None:
    violations = scan_forbidden_imports_in_production()
    assert violations == [], "forbidden execution imports:\n" + "\n".join(violations)
