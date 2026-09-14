# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — shutdown phase readiness."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reliability import ExecutionRuntimeShutdownPhase
from testing_support.execution_operational_readiness.assessment import (
    ReadinessClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_stop_accepting_not_ready() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            shutdown_phase=ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK,
        )
    )
    assert assessment.readiness is ReadinessClassification.NOT_READY
