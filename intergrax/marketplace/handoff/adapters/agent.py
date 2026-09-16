# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent vertical marketplace lifecycle handoff adapter (ME-RB4)."""

from __future__ import annotations

from intergrax.agent_distribution.dynamic_acquisition import DynamicAgentAcquisitionError
from intergrax.contracts.agent_distribution.marketplace_lifecycle_handoff import (
    AGENT_DISTRIBUTION_DOMAIN_AUTHORITY_ID,
    AgentMarketplaceLifecycleHandoffPort,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffStatus,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleHandoffRequest,
)
from intergrax.marketplace.handoff.adapters._ack_outcome import marketplace_outcome_from_ack


class AgentMarketplaceLifecycleHandoffHandler:
    """Delegates marketplace selection to Agent Distribution lifecycle authority."""

    def __init__(self, domain_port: AgentMarketplaceLifecycleHandoffPort) -> None:
        self._domain_port = domain_port

    @property
    def capability_kind(self) -> CapabilityKind:
        return CapabilityKind.AGENT

    @property
    def domain_authority_id(self) -> str:
        return AGENT_DISTRIBUTION_DOMAIN_AUTHORITY_ID

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        payload = request.domain_payload.agent
        if payload is None:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=self.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.INVALID_HANDOFF_REQUEST,
                reason_detail="agent vertical payload is required",
            )

        try:
            ack = self._domain_port.submit_marketplace_lifecycle_handoff(
                payload,
                request_id=request.request_id,
                correlation_id=request.correlation_id,
            )
        except DynamicAgentAcquisitionError as exc:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=self.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.DOMAIN_UNAVAILABLE,
                reason_detail=str(exc),
            )

        return marketplace_outcome_from_ack(
            request_id=request.request_id,
            domain_authority_id=self.domain_authority_id,
            ack=ack,
        )


__all__ = ["AgentMarketplaceLifecycleHandoffHandler"]
