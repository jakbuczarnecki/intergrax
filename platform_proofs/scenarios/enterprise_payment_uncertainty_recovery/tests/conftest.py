"""Shared fixtures for scenario-local application tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.application.references import (
    LabBusinessReferences,
)

_REF = LabBusinessReferences()


@pytest.fixture
def lab_references() -> LabBusinessReferences:
    return _REF


@pytest.fixture
def valid_execution_context() -> ScenarioExecutionContext:
    correlation: Mapping[str, str] = {
        "order_logical_id": _REF.logical_order_id,
        "payment_correlation_id": _REF.payment_intent_reference,
    }
    return ScenarioExecutionContext(
        scenario_id="ERL-QUAL-004",
        scenario_slug="enterprise_payment_uncertainty_recovery",
        variant_id="payment_completed_after_unknown",
        execution_reference="exec-erl-qual-004-lab-0001",
        correlation_ids=correlation,
    )
