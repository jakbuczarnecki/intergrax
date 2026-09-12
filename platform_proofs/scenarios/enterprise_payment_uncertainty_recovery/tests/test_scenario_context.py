"""Scenario execution context validation."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
    validate_scenario_execution_context,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.failures import (
    InvalidScenarioContextError,
)

pytestmark = pytest.mark.unit


def test_validate_accepts_harness_context(valid_execution_context) -> None:
    validate_scenario_execution_context(valid_execution_context)


def test_validate_rejects_empty_variant(valid_execution_context) -> None:
    bad = ScenarioExecutionContext(
        scenario_id=valid_execution_context.scenario_id,
        scenario_slug=valid_execution_context.scenario_slug,
        variant_id="",
        execution_reference=valid_execution_context.execution_reference,
        correlation_ids=valid_execution_context.correlation_ids,
    )
    with pytest.raises(InvalidScenarioContextError):
        validate_scenario_execution_context(bad)


def test_validate_rejects_wrong_qualification_id(valid_execution_context) -> None:
    bad = ScenarioExecutionContext(
        scenario_id="OTHER",
        scenario_slug=valid_execution_context.scenario_slug,
        variant_id=valid_execution_context.variant_id,
        execution_reference=valid_execution_context.execution_reference,
        correlation_ids=valid_execution_context.correlation_ids,
    )
    with pytest.raises(InvalidScenarioContextError):
        validate_scenario_execution_context(bad)
