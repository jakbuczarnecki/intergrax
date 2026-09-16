# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace lifecycle handoff envelope (ME-RB4)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.lifecycle_handoff_intent import (
    MarketplaceLifecycleHandoffIntent,
)
from intergrax.contracts.marketplace.lifecycle_handoff_payloads import (
    AgentLifecycleHandoffPayload,
    SkillLifecycleHandoffPayload,
    ToolLifecycleHandoffPayload,
)

SCHEMA_MARKETPLACE_CAPABILITY_SELECTION_V1: Final = "marketplace_capability_selection.v1"
SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_REQUEST_V1: Final = (
    "marketplace_lifecycle_handoff_request.v1"
)


class MarketplaceCapabilitySelection(BaseModel):
    """Governed marketplace selection — discovery product, not lifecycle mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MARKETPLACE_CAPABILITY_SELECTION_V1
    listing_id: str = Field(min_length=1)
    capability: CapabilityCatalogEntry
    governance_evidence_ref: str | None = None
    provenance_ref: str | None = None


class MarketplaceLifecycleDomainPayload(BaseModel):
    """Exactly one typed vertical payload aligned with ``capability.kind``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: AgentLifecycleHandoffPayload | None = None
    tool: ToolLifecycleHandoffPayload | None = None
    skill: SkillLifecycleHandoffPayload | None = None

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> MarketplaceLifecycleDomainPayload:
        present = sum(
            1
            for item in (self.agent, self.tool, self.skill)
            if item is not None
        )
        if present != 1:
            raise ValueError("exactly one vertical lifecycle handoff payload is required")
        return self

    def capability_kind(self) -> CapabilityKind:
        if self.agent is not None:
            return CapabilityKind.AGENT
        if self.tool is not None:
            return CapabilityKind.TOOL
        if self.skill is not None:
            return CapabilityKind.SKILL
        raise ValueError("no vertical payload present")

    def identity_key(self) -> CapabilityIdentityKey:
        if self.agent is not None:
            return self.agent.capability_identity_key
        if self.tool is not None:
            return self.tool.capability_identity_key
        if self.skill is not None:
            return self.skill.capability_identity_key
        raise ValueError("no vertical payload present")


class MarketplaceLifecycleHandoffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_REQUEST_V1
    request_id: str = Field(min_length=1)
    selection: MarketplaceCapabilitySelection
    intent: MarketplaceLifecycleHandoffIntent
    domain_payload: MarketplaceLifecycleDomainPayload
    correlation_id: str | None = None

    @model_validator(mode="after")
    def _kind_alignment(self) -> MarketplaceLifecycleHandoffRequest:
        selection_kind = self.selection.capability.identity.kind
        payload_kind = self.domain_payload.capability_kind()
        if selection_kind is not payload_kind:
            raise ValueError(
                "selection capability kind must match domain payload kind",
            )
        return self


def selection_identity_key(selection: MarketplaceCapabilitySelection) -> CapabilityIdentityKey:
    return CapabilityIdentityKey.from_discovery_identity(selection.capability.identity)


__all__ = [
    "MarketplaceCapabilitySelection",
    "MarketplaceLifecycleDomainPayload",
    "MarketplaceLifecycleHandoffRequest",
    "SCHEMA_MARKETPLACE_CAPABILITY_SELECTION_V1",
    "SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_REQUEST_V1",
    "selection_identity_key",
]
