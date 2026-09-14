# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — policy/security denials do not mark system unhealthy."""

from __future__ import annotations

import pytest

from testing_support.execution_operational_readiness.assessment import (
    assess_execution_operational_state,
    assessment_unchanged_by_policy_or_security_denial,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_policy_denial_does_not_change_operational_assessment() -> None:
    before = assess_execution_operational_state(baseline_operational_facts())
    after = assess_execution_operational_state(baseline_operational_facts())
    assert assessment_unchanged_by_policy_or_security_denial(before, after)


def test_ee_b4_a_compound_otlp_and_capacity_deterministic() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            best_effort_observability_export_available=False,
            active_root_executions=4,
            capacity_limit=4,
        )
    )
    assert assessment.readiness.name == "NOT_READY"
    assert assessment.health.name == "DEGRADED"
