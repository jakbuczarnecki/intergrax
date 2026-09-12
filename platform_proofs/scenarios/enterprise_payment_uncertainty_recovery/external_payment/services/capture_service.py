"""External acquirer capture processing — persists SoR truth, returns integration outcome."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from uuid import UUID

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.capture import (
    PaymentCaptureCommand,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.integration_result import (
    PaymentProcessingResult,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.persistence import (
    ExternalRealityPersistenceBundle,
    ExternalRealityPersistencePort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.failures import (
    CommunicationFailureKind,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.lifecycle import (
    ExternalPaymentLifecycleState,
    IntegrationResponseState,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.profiles import (
    VariantExecutionProfile,
)

_INTEGRATION_STATE_BY_OUTCOME: dict[str, str] = {
    "unknown": "INDETERMINATE",
    "completed": "CONFIRMED",
    "failed": "DECLINED",
}


class ExternalPaymentCaptureService:
    """Enterprise PSP / acquirer simulator with communication uncertainty."""

    def __init__(self, persistence: ExternalRealityPersistencePort) -> None:
        self._persistence = persistence

    def process_capture(
        self,
        command: PaymentCaptureCommand,
        profile: VariantExecutionProfile,
        *,
        payment_intent_id: UUID,
    ) -> PaymentProcessingResult:
        requested_at = command.request_timestamp
        processing_at = requested_at + timedelta(seconds=2)

        lifecycle, integration_status, failure_kind = self._resolve_channel_outcome(profile)

        effect_id = self._logical_uuid(f"effect:{profile.effect_logical_id}:{profile.variant_id}")
        reality_id = self._logical_uuid(
            f"reality:{profile.effect_logical_id}:{profile.variant_id}"
        )
        integration_state = _INTEGRATION_STATE_BY_OUTCOME.get(
            profile.immediate_integration_outcome,
            "INDETERMINATE",
        )

        bundle = ExternalRealityPersistenceBundle(
            external_payment_effect_id=effect_id,
            external_reality_id=reality_id,
            payment_intent_id=payment_intent_id,
            external_effect_reference=self._external_reference(profile.effect_logical_id),
            correlation_id=command.correlation_id,
            sor_transaction_ref=self._sor_transaction_ref(profile),
            requested_state=profile.business_operation.upper(),
            observed_integration_state=integration_state,
            lifecycle_state=lifecycle,
            sor_truth=profile.sor_truth,
            requested_at=requested_at,
            processed_at=processing_at,
        )
        self._persistence.persist_external_reality(bundle)

        return PaymentProcessingResult(
            correlation_id=command.correlation_id,
            external_reference=bundle.external_effect_reference,
            integration_status=integration_status,
            external_lifecycle_state=lifecycle,
            request_timestamp=requested_at,
            processing_timestamp=processing_at if integration_status != IntegrationResponseState.UNKNOWN else None,
            communication_failure_kind=failure_kind,
        )

    def _resolve_channel_outcome(
        self, profile: VariantExecutionProfile
    ) -> tuple[
        ExternalPaymentLifecycleState,
        IntegrationResponseState,
        CommunicationFailureKind | None,
    ]:
        comm = profile.communication
        if comm.lost_response or profile.immediate_integration_outcome == "unknown":
            return (
                ExternalPaymentLifecycleState.UNKNOWN,
                IntegrationResponseState.UNKNOWN,
                comm.failure_kind,
            )

        if profile.sor_truth.terminal_outcome == "PAYMENT_COMPLETED":
            return (
                ExternalPaymentLifecycleState.COMPLETED,
                IntegrationResponseState.COMPLETED,
                None,
            )
        if profile.sor_truth.terminal_outcome == "PAYMENT_FAILED":
            return (
                ExternalPaymentLifecycleState.FAILED,
                IntegrationResponseState.FAILED,
                None,
            )
        return (
            ExternalPaymentLifecycleState.UNKNOWN,
            IntegrationResponseState.UNKNOWN,
            comm.failure_kind,
        )

    @staticmethod
    def _logical_uuid(logical_id: str) -> UUID:
        return UUID(bytes=hashlib.sha256(f"erl-qual-004:{logical_id}".encode()).digest()[:16])

    @staticmethod
    def _external_reference(effect_logical_id: str) -> str:
        token = re.sub(r"[^a-zA-Z0-9]+", "-", effect_logical_id).strip("-").upper()
        prefix = "EXT"
        return f"{prefix}-{token[-24:]}" if len(token) > 24 else f"{prefix}-{token}"

    @staticmethod
    def _sor_transaction_ref(profile: VariantExecutionProfile) -> str:
        token = f"{profile.effect_logical_id}:{profile.variant_id}"
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", token).strip("-").upper()
        return f"SOR-{slug[-24:]}" if len(slug) > 24 else f"SOR-{slug}"

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(tz=UTC)
