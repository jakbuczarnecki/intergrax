# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.evidence.core_certification_spec import (
    CORE_LEVEL_SCENARIOS,
    CoreCertificationLevel,
    CoreCertificationMode,
    CoreCertificationSurface,
    is_scenario_in_level,
    normalize_core_level,
    scenario_ids_for_level,
)

pytestmark = pytest.mark.unit


def test_core_l1_has_four_scenarios() -> None:
    assert len(CORE_LEVEL_SCENARIOS[CoreCertificationLevel.L1]) == 4
    assert len(scenario_ids_for_level(CoreCertificationLevel.L1)) == 4


def test_core_l2_has_eight_scenarios() -> None:
    assert len(CORE_LEVEL_SCENARIOS[CoreCertificationLevel.L2]) == 8
    assert len(scenario_ids_for_level("CORE-L2")) == 8


def test_core_l3_has_twelve_scenarios() -> None:
    assert len(CORE_LEVEL_SCENARIOS[CoreCertificationLevel.L3]) == 12
    assert len(scenario_ids_for_level("l3")) == 12


def test_normalize_core_level_accepts_enum_and_aliases() -> None:
    assert normalize_core_level(CoreCertificationLevel.L1) is CoreCertificationLevel.L1
    assert normalize_core_level("CORE-L1") is CoreCertificationLevel.L1
    assert normalize_core_level("core-l1") is CoreCertificationLevel.L1
    assert normalize_core_level("l1") is CoreCertificationLevel.L1
    assert normalize_core_level("L2") is CoreCertificationLevel.L2


def test_normalize_core_level_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="invalid core certification level"):
        normalize_core_level("CORE-L9")
    with pytest.raises(ValueError, match="invalid core certification level"):
        normalize_core_level("bogus")


def test_is_scenario_in_level_cumulative() -> None:
    assert is_scenario_in_level("basic_run_completed", CoreCertificationLevel.L3)
    assert is_scenario_in_level("high_risk_tool_hitl", CoreCertificationLevel.L2)
    assert not is_scenario_in_level("high_risk_tool_hitl", CoreCertificationLevel.L1)
    assert is_scenario_in_level(
        "certification_report_emitted",
        CoreCertificationLevel.L1,
    )


def test_certification_surfaces_include_certify_core() -> None:
    assert CoreCertificationSurface.CERTIFY_CORE.value == "certify_core"
    assert CoreCertificationMode.OPERATOR_LOCAL.value == "operator_local"


def test_l2_scenarios_are_superset_of_l1() -> None:
    l1 = set(scenario_ids_for_level(CoreCertificationLevel.L1))
    l2 = set(scenario_ids_for_level(CoreCertificationLevel.L2))
    l3 = set(scenario_ids_for_level(CoreCertificationLevel.L3))
    assert l1 <= l2 <= l3
