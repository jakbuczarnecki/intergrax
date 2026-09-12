"""External boundary lifecycle and variant semantics (data-driven dataset)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.dataset_profile_loader import (
    load_variant_execution_profile,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.in_memory_persistence import (
    InMemoryExternalRealityStore,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.capture import (
    PaymentCaptureCommand,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.lifecycle import (
    ExternalPaymentLifecycleState,
    IntegrationResponseState,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.services.capture_service import (
    ExternalPaymentCaptureService,
)

pytestmark = pytest.mark.unit

_VARIANT_EXPECTATIONS: dict[str, tuple[str, bool]] = {
    "payment_completed_after_unknown": ("PAYMENT_COMPLETED", True),
    "payment_failed_after_unknown": ("PAYMENT_FAILED", False),
    "payment_truth_unavailable": ("TRUTH_INDETERMINATE", False),
}


@pytest.mark.parametrize("variant_id", list(_VARIANT_EXPECTATIONS))
def test_capture_persists_sor_truth_per_variant(variant_id: str) -> None:
    store = InMemoryExternalRealityStore()
    service = ExternalPaymentCaptureService(store)
    profile = load_variant_execution_profile(variant_id)
    command = PaymentCaptureCommand(
        correlation_id="corr-erl-qual-004-0001",
        external_business_reference="PAY-20260912-8F31A",
        idempotency_key="idem-capture-po-2026-004872-v1",
        merchant_order_reference="PO-2026-004872",
        amount=Decimal("12500.00"),
        currency="EUR",
        request_timestamp=datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC),
    )
    result = service.process_capture(
        command,
        profile,
        payment_intent_id=UUID("00000000-0000-4000-8000-000000000001"),
    )

    assert result.integration_status == IntegrationResponseState.UNKNOWN
    assert result.external_lifecycle_state == ExternalPaymentLifecycleState.UNKNOWN
    assert result.communication_failure_kind is not None

    stored = store.latest()
    assert stored is not None
    expected_terminal, expected_captured = _VARIANT_EXPECTATIONS[variant_id]
    assert stored.bundle.sor_truth.terminal_outcome == expected_terminal
    assert stored.bundle.sor_truth.funds_captured is expected_captured


def test_unknown_integration_does_not_imply_payment_failure() -> None:
    profile = load_variant_execution_profile("payment_completed_after_unknown")
    store = InMemoryExternalRealityStore()
    service = ExternalPaymentCaptureService(store)
    command = PaymentCaptureCommand(
        correlation_id="corr-uncertainty",
        external_business_reference="PAY-REF",
        idempotency_key="idem-1",
        merchant_order_reference="PO-1",
        amount=Decimal("10.00"),
        currency="EUR",
        request_timestamp=datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC),
    )
    result = service.process_capture(
        command,
        profile,
        payment_intent_id=UUID("00000000-0000-4000-8000-000000000002"),
    )
    assert result.integration_status == IntegrationResponseState.UNKNOWN
    stored = store.latest()
    assert stored is not None
    assert stored.bundle.sor_truth.terminal_outcome == "PAYMENT_COMPLETED"
