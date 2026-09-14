# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — capacity saturation vs readiness (EE-B1.2)."""

from __future__ import annotations

import pytest

from testing_support.execution_operational_readiness.assessment import (
    HealthClassification,
    LivenessClassification,
    ReadinessClassification,
    SaturationClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_capacity_saturated_not_ready_degraded() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(active_root_executions=4, capacity_limit=4)
    )
    assert assessment.liveness is LivenessClassification.LIVE
    assert assessment.readiness is ReadinessClassification.NOT_READY
    assert assessment.health is HealthClassification.DEGRADED
    assert assessment.saturation is SaturationClassification.SATURATED
