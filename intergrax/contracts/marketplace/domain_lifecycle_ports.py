# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain lifecycle authority ports invoked from marketplace handoff adapters (ME-RB4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.marketplace.lifecycle_handoff_payloads import (
    AgentLifecycleHandoffPayload,
    SkillLifecycleHandoffPayload,
    ToolLifecycleHandoffPayload,
)


class DomainLifecycleHandoffDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class DomainLifecycleHandoffAck(BaseModel):
    """Domain authority acknowledgment — not lifecycle terminal state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: DomainLifecycleHandoffDisposition
    domain_reference: str | None = None
    reason_detail: str = ""


class AgentMarketplaceLifecycleDomainPort(Protocol):
    """Agent Distribution public handoff boundary for marketplace selections."""

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: AgentLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck: ...


class ToolMarketplaceLifecycleDomainPort(Protocol):
    """Tool domain lifecycle handoff boundary (DESIGN GAP: minimal port until canonical tool API)."""

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: ToolLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck: ...


class SkillMarketplaceLifecycleDomainPort(Protocol):
    """Skill domain lifecycle handoff boundary (DESIGN GAP: minimal port until canonical skill API)."""

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: SkillLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck: ...


AGENT_DISTRIBUTION_DOMAIN_AUTHORITY_ID = "agent_distribution"
TOOL_DOMAIN_AUTHORITY_ID = "tool_domain"
SKILL_DOMAIN_AUTHORITY_ID = "skill_domain"


__all__ = [
    "AGENT_DISTRIBUTION_DOMAIN_AUTHORITY_ID",
    "AgentMarketplaceLifecycleDomainPort",
    "DomainLifecycleHandoffAck",
    "DomainLifecycleHandoffDisposition",
    "SKILL_DOMAIN_AUTHORITY_ID",
    "SkillMarketplaceLifecycleDomainPort",
    "TOOL_DOMAIN_AUTHORITY_ID",
    "ToolMarketplaceLifecycleDomainPort",
]
