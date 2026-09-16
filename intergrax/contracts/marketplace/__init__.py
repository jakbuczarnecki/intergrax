# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace product contracts (CAPABILITY-CATALOG-1 Stage 11)."""

from __future__ import annotations

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.marketplace.commercial import (
    SCHEMA_MARKETPLACE_COMMERCIAL_METADATA_V1,
    CommercialModel,
    MarketplaceCommercialMetadata,
)
from intergrax.contracts.marketplace.listing import (
    SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1,
    SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1,
    MarketplaceCapabilityListing,
    MarketplaceCapabilityListingView,
)
from intergrax.contracts.marketplace.listing_projection import MarketplaceListingProjection
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord
from intergrax.contracts.marketplace.metadata_source import MarketplaceMetadataSource
from intergrax.contracts.marketplace.query_context import (
    SCHEMA_MARKETPLACE_QUERY_CONTEXT_V1,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.visibility import (
    SCHEMA_MARKETPLACE_VISIBILITY_V1,
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.contracts.marketplace.publisher import (
    SCHEMA_MARKETPLACE_PUBLISHER_METADATA_V1,
    MarketplacePublisherMetadata,
)
from intergrax.contracts.marketplace.lifecycle_handoff_handler import (
    MarketplaceLifecycleHandoffHandler,
)
from intergrax.contracts.marketplace.lifecycle_handoff_intent import (
    MarketplaceLifecycleHandoffIntent,
)
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffStatus,
    SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_OUTCOME_V1,
)
from intergrax.contracts.marketplace.lifecycle_handoff_payloads import (
    AgentLifecycleHandoffPayload,
    SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1,
    SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1,
    SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1,
    SkillLifecycleHandoffPayload,
    ToolLifecycleHandoffPayload,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceCapabilitySelection,
    MarketplaceLifecycleDomainPayload,
    MarketplaceLifecycleHandoffRequest,
    SCHEMA_MARKETPLACE_CAPABILITY_SELECTION_V1,
    SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_REQUEST_V1,
    selection_identity_key,
)

__all__ = [
    "CommercialModel",
    "MarketplaceCapabilityListing",
    "MarketplaceCapabilityListingView",
    "MarketplaceCommercialMetadata",
    "MarketplaceListingProjection",
    "MarketplaceListingRecord",
    "MarketplaceMetadataSource",
    "MarketplaceQueryContext",
    "MarketplacePublisherMetadata",
    "MarketplaceVisibility",
    "MarketplaceVisibilityScope",
    "SCHEMA_MARKETPLACE_QUERY_CONTEXT_V1",
    "SCHEMA_MARKETPLACE_VISIBILITY_V1",
    "SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1",
    "SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1",
    "SCHEMA_MARKETPLACE_COMMERCIAL_METADATA_V1",
    "SCHEMA_MARKETPLACE_PUBLISHER_METADATA_V1",
    "AgentLifecycleHandoffPayload",
    "DomainLifecycleHandoffAck",
    "DomainLifecycleHandoffDisposition",
    "MarketplaceCapabilitySelection",
    "MarketplaceLifecycleDomainPayload",
    "MarketplaceLifecycleHandoffHandler",
    "MarketplaceLifecycleHandoffIntent",
    "MarketplaceLifecycleHandoffOutcome",
    "MarketplaceLifecycleHandoffReasonCode",
    "MarketplaceLifecycleHandoffRequest",
    "MarketplaceLifecycleHandoffStatus",
    "SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SCHEMA_MARKETPLACE_CAPABILITY_SELECTION_V1",
    "SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_OUTCOME_V1",
    "SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_REQUEST_V1",
    "SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SkillLifecycleHandoffPayload",
    "ToolLifecycleHandoffPayload",
    "selection_identity_key",
]
