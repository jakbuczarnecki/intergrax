"""Business workflow service behavior."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.failures import (
    MissingBusinessEntityError,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.observability import (
    RecordingScenarioApplicationObservability,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.services.enterprise_payment_workflow import (
    EnterprisePaymentWorkflowService,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.tests.support.lab_ports import (
    LabReferenceOrderAccess,
    LabReferencePaymentWorkflow,
)

pytestmark = pytest.mark.unit


def test_workflow_loads_order_and_requests_payment(valid_execution_context, lab_references) -> None:
    service = EnterprisePaymentWorkflowService(
        order_access=LabReferenceOrderAccess(lab_references),
        payment_workflow=LabReferencePaymentWorkflow(lab_references),
        observability=RecordingScenarioApplicationObservability(),
    )
    outcome = service.execute(valid_execution_context)

    assert outcome.order.organization.legal_name == lab_references.organization_legal_name
    assert outcome.payment_request.related_order_number == lab_references.order_number


def test_workflow_surfaces_missing_order(valid_execution_context, lab_references) -> None:
    context = valid_execution_context
    bad_correlation = dict(context.correlation_ids)
    bad_correlation["order_logical_id"] = "erl-qual-004-ord-missing"
    from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
        ScenarioExecutionContext,
    )

    missing_context = ScenarioExecutionContext(
        scenario_id=context.scenario_id,
        scenario_slug=context.scenario_slug,
        variant_id=context.variant_id,
        execution_reference=context.execution_reference,
        correlation_ids=bad_correlation,
    )
    service = EnterprisePaymentWorkflowService(
        order_access=LabReferenceOrderAccess(lab_references),
        payment_workflow=LabReferencePaymentWorkflow(lab_references),
        observability=RecordingScenarioApplicationObservability(),
    )
    with pytest.raises(MissingBusinessEntityError):
        service.execute(missing_context)
