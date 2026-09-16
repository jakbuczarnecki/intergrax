# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deliver typed handoff envelopes to downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumer,
    CapabilityHandoffConsumerError,
    CapabilityHandoffDeliveryAdmission,
    CapabilityHandoffDeliveryAdmissionError,
    CapabilityHandoffDeliveryAdmissionVerdict,
    CapabilityHandoffDeliveryDisposition,
    CapabilityHandoffDeliveryLifecycleTransitionError,
    CapabilityHandoffDeliveryOutcomeUncertainError,
    CapabilityHandoffDeliveryResult,
    CapabilityHandoffEnvelope,
    CapabilityHandoffIdentityConflictError,
    CapabilityHandoffTraceEvidenceConsumer,
)
from intergrax.marketplace.handoff_traceability.errors import (
    MarketplaceHandoffConsumerIdentityMismatchError,
    MarketplaceHandoffTenantConsistencyError,
)


@dataclass(frozen=True, slots=True)
class CapabilityHandoffDeliveryService:
    """Routes envelopes to a structural consumer with mandatory delivery lifecycle admission."""

    consumer: CapabilityHandoffConsumer
    delivery_admission: CapabilityHandoffDeliveryAdmission
    trace_evidence_consumer: CapabilityHandoffTraceEvidenceConsumer | None = None

    def deliver(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryResult:
        self._validate_tenant_consistency(envelope)
        self._validate_consumer_identity(envelope)
        actual_consumer_id = self.consumer.consumer_id
        try:
            reservation = self.delivery_admission.reserve(envelope)
        except CapabilityHandoffIdentityConflictError:
            raise
        except CapabilityHandoffDeliveryAdmissionError:
            raise
        except Exception as exc:
            raise CapabilityHandoffDeliveryAdmissionError(str(exc)) from exc

        if (
            reservation.verdict
            is CapabilityHandoffDeliveryAdmissionVerdict.ALREADY_DELIVERED_IDENTICAL
        ):
            return CapabilityHandoffDeliveryResult(
                disposition=CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED,
                handoff_id=envelope.handoff_id,
                downstream_consumer_id=actual_consumer_id,
            )
        if reservation.verdict is CapabilityHandoffDeliveryAdmissionVerdict.IN_PROGRESS_IDENTICAL:
            return CapabilityHandoffDeliveryResult(
                disposition=CapabilityHandoffDeliveryDisposition.IN_PROGRESS_SKIPPED,
                handoff_id=envelope.handoff_id,
                downstream_consumer_id=actual_consumer_id,
            )

        try:
            self.consumer.consume(envelope)
        except CapabilityHandoffConsumerError:
            self._release_after_consumer_failure(envelope.handoff_id)
            raise
        except Exception as exc:
            self._release_after_consumer_failure(envelope.handoff_id)
            raise CapabilityHandoffConsumerError(str(exc)) from exc

        try:
            self.delivery_admission.mark_delivered(envelope.handoff_id)
        except CapabilityHandoffDeliveryAdmissionError as exc:
            raise CapabilityHandoffDeliveryOutcomeUncertainError(
                "downstream consumer succeeded but delivery lifecycle could not be committed",
            ) from exc
        except Exception as exc:
            raise CapabilityHandoffDeliveryOutcomeUncertainError(str(exc)) from exc

        self._record_trace_observation(envelope)
        return CapabilityHandoffDeliveryResult(
            disposition=CapabilityHandoffDeliveryDisposition.DELIVERED,
            handoff_id=envelope.handoff_id,
            downstream_consumer_id=actual_consumer_id,
        )

    def _release_after_consumer_failure(self, handoff_id: str) -> None:
        try:
            self.delivery_admission.mark_delivery_failed(handoff_id)
        except CapabilityHandoffDeliveryAdmissionError:
            raise
        except Exception as exc:
            raise CapabilityHandoffDeliveryLifecycleTransitionError(str(exc)) from exc

    def _record_trace_observation(self, envelope: CapabilityHandoffEnvelope) -> None:
        if self.trace_evidence_consumer is None:
            return
        try:
            self.trace_evidence_consumer.record_handoff(envelope)
        except Exception:
            return

    def _validate_consumer_identity(self, envelope: CapabilityHandoffEnvelope) -> None:
        actual = self.consumer.consumer_id
        if envelope.downstream_consumer_id != actual:
            raise MarketplaceHandoffConsumerIdentityMismatchError(
                "envelope downstream_consumer_id must equal delivery consumer consumer_id",
            )

    @staticmethod
    def _validate_tenant_consistency(envelope: CapabilityHandoffEnvelope) -> None:
        ctx = envelope.discovery_trace.marketplace_query_context
        if envelope.tenant_id != ctx.tenant_id:
            raise MarketplaceHandoffTenantConsistencyError(
                "handoff tenant_id must equal marketplace_query_context.tenant_id",
            )


__all__ = ["CapabilityHandoffDeliveryService"]
