# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge ME-10 handoff envelope into ME-RB4 lifecycle handoff requests."""

from __future__ import annotations

from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffEnvelope
from intergrax.contracts.marketplace.lifecycle_handoff_intent import (
    MarketplaceLifecycleHandoffIntent,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleDomainPayload,
    MarketplaceLifecycleHandoffRequest,
    marketplace_capability_selection,
)


def lifecycle_handoff_request_from_envelope(
    envelope: CapabilityHandoffEnvelope,
    *,
    request_id: str,
    intent: MarketplaceLifecycleHandoffIntent,
    domain_payload: MarketplaceLifecycleDomainPayload,
    correlation_id: str | None = None,
) -> MarketplaceLifecycleHandoffRequest:
    """Map neutral traceability envelope into domain lifecycle handoff request."""
    selection = envelope.explicit_selection
    if selection.listing_id is None:
        raise ValueError("explicit_selection.listing_id is required for lifecycle handoff")
    capability_entry = envelope.selected_release
    from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry

    catalog_entry = CapabilityCatalogEntry(
        identity=capability_entry.discovery,
        provenance=capability_entry.to_provenance(),
        display_label=capability_entry.discovery.logical.logical_id,
    )
    marketplace_selection = marketplace_capability_selection(
        listing_id=selection.listing_id,
        capability=catalog_entry,
        governance_evidence_ref=selection.governance_evidence_ref,
        provenance_ref=None,
    )
    if marketplace_selection.selected_release != envelope.selected_release:
        raise ValueError("lifecycle selection release must match envelope selected_release")
    return MarketplaceLifecycleHandoffRequest(
        request_id=request_id,
        selection=marketplace_selection,
        intent=intent,
        domain_payload=domain_payload,
        correlation_id=correlation_id or envelope.discovery_correlation_id,
    )


__all__ = ["lifecycle_handoff_request_from_envelope"]
