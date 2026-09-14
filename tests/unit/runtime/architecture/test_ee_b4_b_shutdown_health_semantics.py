# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — readiness/liveness mapping during shutdown (EE-B4-A reuse)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reliability import ExecutionRuntimeShutdownPhase
from testing_support.execution_operational_readiness.assessment import (
    LivenessClassification,
    ReadinessClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_b_readiness_false_after_stop() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            shutdown_phase=ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK,
        )
    )
    assert assessment.readiness is ReadinessClassification.NOT_READY


def test_ee_b4_b_liveness_live_during_drain() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            shutdown_phase=ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS,
        )
    )
    assert assessment.liveness is LivenessClassification.LIVE


def test_ee_b4_b_liveness_false_after_terminate() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            shutdown_phase=ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS,
        )
    )
    assert assessment.liveness is LivenessClassification.NOT_LIVE
