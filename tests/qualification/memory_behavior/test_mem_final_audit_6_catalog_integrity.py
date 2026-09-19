# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 catalog integrity gates."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.qualification.memory_behavior.catalog import MEM_AUDIT_6_SCENARIOS
from tests.qualification.memory_behavior.contracts import BehaviorScenarioCategory

pytestmark = pytest.mark.gate

_MINIMUM_BY_CATEGORY = {
    BehaviorScenarioCategory.USER: 15,
    BehaviorScenarioCategory.SESSION: 4,
    BehaviorScenarioCategory.TASK: 4,
    BehaviorScenarioCategory.PROJECTION_LIFECYCLE: 5,
    BehaviorScenarioCategory.SECURITY: 6,
}


def test_mem_audit_6_catalog_minimum_coverage() -> None:
    counts = Counter(ref.category for ref in MEM_AUDIT_6_SCENARIOS)
    for category, minimum in _MINIMUM_BY_CATEGORY.items():
        assert counts[category] >= minimum, f"{category}: {counts[category]} < {minimum}"


def test_mem_audit_6_scenario_ids_unique() -> None:
    ids = [ref.scenario_id for ref in MEM_AUDIT_6_SCENARIOS]
    assert len(ids) == len(set(ids))
