# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool vertical marketplace lifecycle handoff adapter (ME-RB4)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.domain_lifecycle_ports import (
    TOOL_DOMAIN_AUTHORITY_ID,
    DomainLifecycleHandoffDisposition,
    ToolMarketplaceLifecycleDomainPort,
)
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffStatus,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleHandoffRequest,
)


class ToolMarketplaceLifecycleHandoffHandler:
    """Delegates marketplace selection to Tool domain lifecycle authority."""

    def __init__(self, domain_port: ToolMarketplaceLifecycleDomainPort) -> None:
        self._domain_port = domain_port

    @property
    def capability_kind(self) -> CapabilityKind:
        return CapabilityKind.TOOL

    @property
    def domain_authority_id(self) -> str:
        return TOOL_DOMAIN_AUTHORITY_ID

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        payload = request.domain_payload.tool
        if payload is None:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=self.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.INVALID_HANDOFF_REQUEST,
                reason_detail="tool vertical payload is required",
            )

        try:
            ack = self._domain_port.submit_marketplace_lifecycle_handoff(
                payload,
                request_id=request.request_id,
                correlation_id=request.correlation_id,
            )
        except Exception as exc:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.REJECTED,
                domain_authority_id=self.domain_authority_id,
                reason_code=MarketplaceLifecycleHandoffReasonCode.DOMAIN_UNAVAILABLE,
                reason_detail=str(exc),
            )

        if ack.disposition is DomainLifecycleHandoffDisposition.ACCEPTED:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.ACCEPTED,
                domain_authority_id=self.domain_authority_id,
                domain_reference=ack.domain_reference,
                reason_detail=ack.reason_detail,
            )
        if ack.disposition is DomainLifecycleHandoffDisposition.DEFERRED:
            return MarketplaceLifecycleHandoffOutcome(
                request_id=request.request_id,
                status=MarketplaceLifecycleHandoffStatus.DEFERRED,
                domain_authority_id=self.domain_authority_id,
                domain_reference=ack.domain_reference,
                reason_detail=ack.reason_detail,
            )
        return MarketplaceLifecycleHandoffOutcome(
            request_id=request.request_id,
            status=MarketplaceLifecycleHandoffStatus.REJECTED,
            domain_authority_id=self.domain_authority_id,
            reason_code=MarketplaceLifecycleHandoffReasonCode.DOMAIN_REJECTED,
            domain_reference=ack.domain_reference,
            reason_detail=ack.reason_detail,
        )


__all__ = ["ToolMarketplaceLifecycleHandoffHandler"]
