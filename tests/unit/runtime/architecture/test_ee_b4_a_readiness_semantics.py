# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — readiness semantics."""

from __future__ import annotations

import pytest

from testing_support.execution_operational_readiness.assessment import (
    ReadinessClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_normal_state_ready() -> None:
    assessment = assess_execution_operational_state(baseline_operational_facts())
    assert assessment.readiness is ReadinessClassification.READY


def test_ee_b4_a_startup_not_ready() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(startup_complete=False)
    )
    assert assessment.readiness is ReadinessClassification.STARTING
