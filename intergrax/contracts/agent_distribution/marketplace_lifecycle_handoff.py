# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent Distribution marketplace lifecycle handoff boundary (ME-RB4-C1)."""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffAck

SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1: Final = "agent_lifecycle_handoff_payload.v1"

AGENT_DISTRIBUTION_DOMAIN_AUTHORITY_ID = "agent_distribution"


class AgentLifecycleHandoffPayload(BaseModel):
    """Marketplace→Agent Distribution handoff slice; AC-3/AC-4 request built by domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1
    operation_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    application_environment_id: str = Field(min_length=1)
    catalog_entry_id: str = Field(min_length=1)
    capability_identity_key: CapabilityIdentityKey


class AgentMarketplaceLifecycleHandoffPort(Protocol):
    """Agent Distribution-owned handoff boundary consumed by marketplace adapters."""

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: AgentLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck: ...


__all__ = [
    "AGENT_DISTRIBUTION_DOMAIN_AUTHORITY_ID",
    "AgentLifecycleHandoffPayload",
    "AgentMarketplaceLifecycleHandoffPort",
    "SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1",
]
