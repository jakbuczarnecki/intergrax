# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — liveness semantics."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reliability import ExecutionRuntimeShutdownPhase
from testing_support.execution_operational_readiness.assessment import (
    LivenessClassification,
    assess_execution_operational_state,
)
from tests.unit.runtime.architecture._ee_b4_a_facts import baseline_operational_facts

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b4_a_normal_state_live() -> None:
    assessment = assess_execution_operational_state(baseline_operational_facts())
    assert assessment.liveness is LivenessClassification.LIVE


def test_ee_b4_a_drain_may_remain_live() -> None:
    assessment = assess_execution_operational_state(
        baseline_operational_facts(
            shutdown_phase=ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS,
        )
    )
    assert assessment.liveness is LivenessClassification.LIVE
