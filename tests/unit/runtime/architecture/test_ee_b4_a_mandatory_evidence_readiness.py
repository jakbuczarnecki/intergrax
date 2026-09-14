# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — mandatory evidence persistence vs readiness."""

from __future__ import annotations

import pytest

from testing_support.execution_operational_readiness.assessment import (
    HealthClassification,
    ReadinessClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_mandatory_evidence_unavailable_not_ready() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(mandatory_evidence_persistence_available=False)
    )
    assert assessment.readiness is ReadinessClassification.NOT_READY
    assert assessment.health is HealthClassification.UNHEALTHY
