# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deliver typed handoff envelopes to downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumer,
    CapabilityHandoffConsumerError,
    CapabilityHandoffDeliveryDisposition,
    CapabilityHandoffDeliveryResult,
    CapabilityHandoffEnvelope,
    CapabilityHandoffTraceEvidenceConsumer,
)
from intergrax.marketplace.handoff_traceability.errors import (
    MarketplaceHandoffTenantConsistencyError,
)


@dataclass(frozen=True, slots=True)
class CapabilityHandoffDeliveryService:
    """Routes envelopes to a structural consumer; optional trace evidence is deduped by handoff_id."""

    consumer: CapabilityHandoffConsumer
    trace_evidence_consumer: CapabilityHandoffTraceEvidenceConsumer | None = None

    def deliver(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryResult:
        self._validate_tenant_consistency(envelope)
        duplicate = False
        if self.trace_evidence_consumer is not None:
            recorded = self.trace_evidence_consumer.record_handoff(envelope)
            duplicate = not recorded
        if duplicate:
            return CapabilityHandoffDeliveryResult(
                disposition=CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED,
                handoff_id=envelope.handoff_id,
                downstream_consumer_id=envelope.downstream_consumer_id,
            )
        try:
            self.consumer.consume(envelope)
        except CapabilityHandoffConsumerError:
            raise
        except Exception as exc:
            raise CapabilityHandoffConsumerError(str(exc)) from exc
        return CapabilityHandoffDeliveryResult(
            disposition=CapabilityHandoffDeliveryDisposition.DELIVERED,
            handoff_id=envelope.handoff_id,
            downstream_consumer_id=envelope.downstream_consumer_id,
        )

    @staticmethod
    def _validate_tenant_consistency(envelope: CapabilityHandoffEnvelope) -> None:
        ctx = envelope.discovery_trace.marketplace_query_context
        if envelope.tenant_id is None:
            return
        if ctx.tenant_id is not None and envelope.tenant_id != ctx.tenant_id:
            raise MarketplaceHandoffTenantConsistencyError(
                "handoff tenant_id must equal marketplace_query_context.tenant_id",
            )


__all__ = ["CapabilityHandoffDeliveryService"]
