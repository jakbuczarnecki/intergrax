# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.readiness.section_40_12_checklist import (
    REFERENCE_MUTATING_CAPABILITY,
    build_section_40_12_reference_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_section_40_12_reference_checklist_passes() -> None:
    report = build_section_40_12_reference_report()
    assert report.reference_capability == REFERENCE_MUTATING_CAPABILITY
    assert report.all_passed is True
    item_ids = {item.item_id for item in report.items}
    assert "40.1" in item_ids
    assert "40.2" in item_ids
    assert all(item.status.value in {"pass", "not_applicable"} for item in report.items)
