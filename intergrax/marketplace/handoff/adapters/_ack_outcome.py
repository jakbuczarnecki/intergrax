# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map neutral domain handoff ack to marketplace outcome (ME-RB4-C1)."""

from __future__ import annotations

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffStatus,
)


def marketplace_outcome_from_ack(
    *,
    request_id: str,
    domain_authority_id: str,
    ack: DomainLifecycleHandoffAck,
) -> MarketplaceLifecycleHandoffOutcome:
    if ack.disposition is DomainLifecycleHandoffDisposition.ACCEPTED:
        return MarketplaceLifecycleHandoffOutcome(
            request_id=request_id,
            status=MarketplaceLifecycleHandoffStatus.ACCEPTED,
            domain_authority_id=domain_authority_id,
            domain_reference=ack.domain_reference,
            reason_detail=ack.reason_detail,
        )
    if ack.disposition is DomainLifecycleHandoffDisposition.DEFERRED:
        return MarketplaceLifecycleHandoffOutcome(
            request_id=request_id,
            status=MarketplaceLifecycleHandoffStatus.DEFERRED,
            domain_authority_id=domain_authority_id,
            domain_reference=ack.domain_reference,
            reason_detail=ack.reason_detail,
        )
    return MarketplaceLifecycleHandoffOutcome(
        request_id=request_id,
        status=MarketplaceLifecycleHandoffStatus.REJECTED,
        domain_authority_id=domain_authority_id,
        reason_code=MarketplaceLifecycleHandoffReasonCode.DOMAIN_REJECTED,
        domain_reference=ack.domain_reference,
        reason_detail=ack.reason_detail,
    )


__all__ = ["marketplace_outcome_from_ack"]
