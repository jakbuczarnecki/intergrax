# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — best-effort observability export degradation (EE-B2 alignment)."""

from __future__ import annotations

import pytest

from testing_support.execution_operational_readiness.assessment import (
    HealthClassification,
    LivenessClassification,
    ReadinessClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_otlp_failure_live_ready_degraded() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(best_effort_observability_export_available=False)
    )
    assert assessment.liveness is LivenessClassification.LIVE
    assert assessment.readiness is ReadinessClassification.READY
    assert assessment.health is HealthClassification.DEGRADED
