# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — health classification model."""

from __future__ import annotations

import pytest

from testing_support.execution_operational_readiness.assessment import (
    HealthClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_normal_state_healthy_ready_live() -> None:
    from testing_support.execution_operational_readiness.assessment import (
        LivenessClassification,
        ReadinessClassification,
    )

    assessment = assess_execution_operational_state(baseline_operational_facts())
    assert assessment.health is HealthClassification.HEALTHY
    assert assessment.readiness is ReadinessClassification.READY
    assert assessment.liveness is LivenessClassification.LIVE


def test_ee_b4_a_health_precedence_unhealthy_over_degraded() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            mandatory_evidence_persistence_available=False,
            best_effort_observability_export_available=False,
            active_root_executions=4,
            capacity_limit=4,
        )
    )
    assert assessment.health is HealthClassification.UNHEALTHY
