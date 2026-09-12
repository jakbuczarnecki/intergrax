"""Application composition root startup and orchestration."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.dependencies import (
    ApplicationDependencies,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.root import (
    ScenarioApplicationCompositionRoot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.workflow import (
    BusinessWorkflowPhase,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.observability import (
    BusinessActionKind,
    RecordingScenarioApplicationObservability,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.tests.support.lab_ports import (
    LabReferenceOrderAccess,
    LabReferencePaymentWorkflow,
)

pytestmark = pytest.mark.unit


def test_composition_root_runs_workflow_and_records_observability(
    valid_execution_context,
    lab_references,
) -> None:
    observability = RecordingScenarioApplicationObservability()
    root = ScenarioApplicationCompositionRoot(
        ApplicationDependencies(
            order_access=LabReferenceOrderAccess(lab_references),
            payment_workflow=LabReferencePaymentWorkflow(lab_references),
            observability=observability,
        )
    )
    result = root.run(valid_execution_context)

    assert result.workflow_outcome.phase is BusinessWorkflowPhase.PAYMENT_REQUESTED
    assert result.workflow_outcome.order.order_number == lab_references.order_number
    assert (
        result.workflow_outcome.payment_request.intent_reference
        == lab_references.payment_intent_reference
    )
    assert len(observability.started) == 1
    assert len(observability.prepared) == 1
    action_kinds = [entry[1] for entry in observability.actions]
    assert action_kinds == [
        BusinessActionKind.ORDER_LOADED,
        BusinessActionKind.PAYMENT_CAPTURE_REQUESTED,
    ]
