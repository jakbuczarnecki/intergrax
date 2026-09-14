# © Artur Czarnecki. All rights reserved.

"""EE-B1.2 — capacity contract (typed, frozen, provider-neutral)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_capacity import (
    ExecutionCapacityAdmissionDecision,
    ExecutionCapacityAssessmentContext,
    RootExecutionCapacityEvaluator,
    assess_root_execution_capacity,
)
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b1_2_admission_decision_enum_stable() -> None:
    assert tuple(ExecutionCapacityAdmissionDecision) == ("ALLOW", "DEFER", "REJECT")


def test_ee_b1_2_assess_allow_when_under_limit() -> None:
    ctx = ExecutionCapacityAssessmentContext(
        active_root_executions=1,
        capacity_limit=3,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    assert (
        assess_root_execution_capacity(ctx) is ExecutionCapacityAdmissionDecision.ALLOW
    )


def test_ee_b1_2_assess_reject_when_saturated_reject_mode() -> None:
    ctx = ExecutionCapacityAssessmentContext(
        active_root_executions=2,
        capacity_limit=2,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    assert (
        assess_root_execution_capacity(ctx) is ExecutionCapacityAdmissionDecision.REJECT
    )


def test_ee_b1_2_assess_defer_when_saturated_wait_mode() -> None:
    ctx = ExecutionCapacityAssessmentContext(
        active_root_executions=2,
        capacity_limit=2,
        overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
    )
    assert (
        assess_root_execution_capacity(ctx) is ExecutionCapacityAdmissionDecision.DEFER
    )


def test_ee_b1_2_evaluator_protocol_instance() -> None:
    evaluator = RootExecutionCapacityEvaluator()
    ctx = ExecutionCapacityAssessmentContext(
        active_root_executions=0,
        capacity_limit=1,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    assert evaluator.evaluate(ctx) is ExecutionCapacityAdmissionDecision.ALLOW


def test_ee_b1_2_assessment_context_rejects_invalid_counters() -> None:
    with pytest.raises(ValueError, match="active_root_executions"):
        ExecutionCapacityAssessmentContext(
            active_root_executions=-1,
            capacity_limit=1,
            overload_mode=ExecutionCapacityOverloadMode.REJECT,
        )
    with pytest.raises(ValueError, match="capacity_limit"):
        ExecutionCapacityAssessmentContext(
            active_root_executions=0,
            capacity_limit=0,
            overload_mode=ExecutionCapacityOverloadMode.REJECT,
        )


def test_ee_b1_2_execution_capacity_policy_frozen_forbid() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=2)
    assert policy.model_config.get("frozen") is True
    assert policy.model_config.get("extra") == "forbid"
    with pytest.raises(ValidationError):
        ExecutionCapacityPolicy(max_concurrent_root_executions=2, extra_field=1)
