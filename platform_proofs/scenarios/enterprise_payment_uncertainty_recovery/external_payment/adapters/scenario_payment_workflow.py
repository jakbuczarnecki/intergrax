"""Application PaymentWorkflowPort backed by the external payment boundary."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.entities import (
    OrderSnapshot,
    PaymentCaptureRequest,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.dataset_profile_loader import (
    load_variant_execution_profile,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.capture import (
    PaymentCaptureCommand,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.services.capture_service import (
    ExternalPaymentCaptureService,
)


class ScenarioExternalPaymentWorkflow:
    """Routes application capture requests through the external acquirer simulator."""

    def __init__(self, capture_service: ExternalPaymentCaptureService) -> None:
        self._capture_service = capture_service

    def request_capture(
        self,
        context: ScenarioExecutionContext,
        order: OrderSnapshot,
    ) -> PaymentCaptureRequest:
        profile = load_variant_execution_profile(context.variant_id)
        correlation_id = context.correlation_ids.get(
            "payment_correlation_id",
            profile.effect_logical_id,
        )
        intent_reference = correlation_id
        command = PaymentCaptureCommand(
            correlation_id=correlation_id,
            external_business_reference=intent_reference,
            idempotency_key=f"idem-{order.order_number}-v1",
            merchant_order_reference=order.order_number,
            amount=order.amount,
            currency=order.currency,
            request_timestamp=datetime.now(tz=UTC),
        )
        payment_intent_id = self._payment_intent_uuid(profile.effect_logical_id)
        self._capture_service.process_capture(
            command,
            profile,
            payment_intent_id=payment_intent_id,
        )
        return PaymentCaptureRequest(
            intent_reference=intent_reference,
            idempotency_key=command.idempotency_key,
            related_order_number=order.order_number,
            amount=order.amount,
            currency=order.currency,
        )

    @staticmethod
    def _payment_intent_uuid(effect_logical_id: str) -> UUID:
        return UUID(
            bytes=hashlib.sha256(f"erl-qual-004:intent:{effect_logical_id}".encode()).digest()[:16]
        )
