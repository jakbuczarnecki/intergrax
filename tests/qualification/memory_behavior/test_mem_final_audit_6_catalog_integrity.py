# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 catalog integrity gates."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.qualification.memory_behavior.behavior_registry import MEM_AUDIT_6_BEHAVIOR_CASES
from tests.qualification.memory_behavior.catalog import MEM_AUDIT_6_SCENARIOS
from tests.qualification.memory_behavior.contracts import BehaviorScenarioCategory

pytestmark = pytest.mark.gate

_MINIMUM_BY_CATEGORY = {
    BehaviorScenarioCategory.USER: 15,
    BehaviorScenarioCategory.SESSION: 4,
    BehaviorScenarioCategory.TASK: 4,
    BehaviorScenarioCategory.PROJECTION_LIFECYCLE: 5,
    BehaviorScenarioCategory.SECURITY: 5,
    BehaviorScenarioCategory.HARNESS_INTEGRITY: 1,
}

_BEHAVIORAL_CATALOG_CATEGORIES = frozenset(
    {
        BehaviorScenarioCategory.USER,
        BehaviorScenarioCategory.SESSION,
        BehaviorScenarioCategory.TASK,
        BehaviorScenarioCategory.PROJECTION_LIFECYCLE,
        BehaviorScenarioCategory.SECURITY,
    }
)


def test_mem_audit_6_catalog_minimum_coverage() -> None:
    counts = Counter(ref.category for ref in MEM_AUDIT_6_SCENARIOS)
    for category, minimum in _MINIMUM_BY_CATEGORY.items():
        assert counts[category] >= minimum, f"{category}: {counts[category]} < {minimum}"


def test_mem_audit_6_scenario_ids_unique() -> None:
    ids = [ref.scenario_id for ref in MEM_AUDIT_6_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_mem_audit_6_catalog_matches_behavior_registry() -> None:
    catalog_ids = {
        ref.scenario_id
        for ref in MEM_AUDIT_6_SCENARIOS
        if ref.category in _BEHAVIORAL_CATALOG_CATEGORIES
    }
    registry_ids = {case.scenario_id for case in MEM_AUDIT_6_BEHAVIOR_CASES}
    assert catalog_ids == registry_ids
