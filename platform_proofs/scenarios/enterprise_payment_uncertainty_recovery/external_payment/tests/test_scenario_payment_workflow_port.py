"""PaymentWorkflowPort wiring through the external boundary."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.services.enterprise_payment_workflow import (
    EnterprisePaymentWorkflowService,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.observability import (
    RecordingScenarioApplicationObservability,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.in_memory_persistence import (
    InMemoryExternalRealityStore,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.scenario_payment_workflow import (
    ScenarioExternalPaymentWorkflow,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.services.capture_service import (
    ExternalPaymentCaptureService,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.tests.support.lab_ports import (
    LabReferenceOrderAccess,
)

pytestmark = pytest.mark.unit


def test_application_workflow_uses_external_boundary(valid_execution_context, lab_references) -> None:
    store = InMemoryExternalRealityStore()
    capture = ExternalPaymentCaptureService(store)
    payment_port = ScenarioExternalPaymentWorkflow(capture)
    service = EnterprisePaymentWorkflowService(
        order_access=LabReferenceOrderAccess(lab_references),
        payment_workflow=payment_port,
        observability=RecordingScenarioApplicationObservability(),
    )
    outcome = service.execute(valid_execution_context)
    assert outcome.payment_request.intent_reference == lab_references.payment_intent_reference
    assert store.latest() is not None
