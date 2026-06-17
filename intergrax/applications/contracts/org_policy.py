# © Artur Czarnecki. All rights reserved.

"""Organizational policy envelope — Tier-3 virtual workforce contract (architecture §39 · ACP-ORG-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.environment_profile.sub_profiles import (
    GuardrailProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode


class ScenarioBinding(BaseModel):
    """Map trigger (capability or metadata key) to required playbook."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    trigger: str | list[str]
    required_playbook_id: str
    mandatory: bool = True


class ChannelPolicy(BaseModel):
    """Allowed and denied communication channels for the org."""

    model_config = ConfigDict(extra="forbid")

    allowed_channels: list[str] = Field(default_factory=list)
    denied_channels: list[str] = Field(default_factory=list)
    default_channel: str | None = None


class CommunicationRules(BaseModel):
    """Prompt overlay and conduct constraints."""

    model_config = ConfigDict(extra="forbid")

    required_disclosures: list[str] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(default_factory=list)
    tone: str | None = None
    locale_default: str | None = None


class ToolPolicyOverlay(BaseModel):
    """Org-wide tool deny/allow patterns on top of agent ToolProfile."""

    model_config = ConfigDict(extra="forbid")

    deny_patterns: list[str] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)


class OrganizationalPolicyEnvelope(BaseModel):
    """
    One simulated organization / tenant policy surface (ACP-ORG-1).

    Attached to ``ApplicationEnvironmentProfile.organizational_policy``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "org_policy_envelope.v1"
    organization_id: str
    display_name: str
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    policy_rules: PolicyRulesProfile | None = None
    guardrails: GuardrailProfile | None = None
    sop_catalog_path: Path | None = None
    scenario_bindings: list[ScenarioBinding] = Field(default_factory=list)
    rag_playbook_collection: str | None = None
    channel_policy: ChannelPolicy = Field(default_factory=ChannelPolicy)
    tool_policy_overlay: ToolPolicyOverlay | None = None
    communication_rules: CommunicationRules = Field(default_factory=CommunicationRules)
    compliance_profile_id: str | None = None
    observability_labels: dict[str, str] = Field(default_factory=dict)


class OrganizationalPolicyContext(BaseModel):
    """Runtime merge of envelope + roster role (ACP-ORG-2)."""

    model_config = ConfigDict(extra="forbid")

    organization_id: str
    org_role_id: str | None = None
    active_scenario_id: str | None = None
    active_playbook_ids: list[str] = Field(default_factory=list)
    channel_policy: ChannelPolicy = Field(default_factory=ChannelPolicy)
    effective_tool_denies: list[str] = Field(default_factory=list)
    prompt_overlay_ids: list[str] = Field(default_factory=list)
    observability_labels: dict[str, str] = Field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    domain_fragments: dict[str, Any] = Field(default_factory=dict)


def product_host_org_envelope(
    *,
    product_id: str,
    display_name: str | None = None,
    primary_capability: str | None = None,
) -> OrganizationalPolicyEnvelope:
    """UC-11 org envelope for Tier-3 product hosts (ACP-CLOSE-ORG-2)."""
    scenario_bindings: list[ScenarioBinding] = []
    if primary_capability:
        scenario_bindings.append(
            ScenarioBinding(
                scenario_id=f"{product_id}_primary",
                trigger=primary_capability,
                required_playbook_id=f"sop.{product_id}.primary",
            ),
        )
    return OrganizationalPolicyEnvelope(
        organization_id=f"{product_id}.org",
        display_name=display_name or f"{product_id.replace('_', ' ').title()} Organization",
        execution_mode=ExecutionMode.STRICT,
        channel_policy=ChannelPolicy(
            allowed_channels=["chat", "ticket", "email"],
            denied_channels=["phone", "sms"],
            default_channel="chat",
        ),
        tool_policy_overlay=ToolPolicyOverlay(
            deny_patterns=["phone.*", "sms.*"],
        ),
        communication_rules=CommunicationRules(
            required_disclosures=[f"org.disclosure.{product_id}"],
            tone="formal",
        ),
        scenario_bindings=scenario_bindings,
        compliance_profile_id=f"{product_id}.org.compliance.v1",
        observability_labels={"org": product_id, "host": "product"},
    )


def lab_strict_org_envelope(
    *,
    organization_id: str = "lab.virtual_org",
) -> OrganizationalPolicyEnvelope:
    """Reference strict org fixture for UC-11 / ACP-ORG-5 eval."""
    return OrganizationalPolicyEnvelope(
        organization_id=organization_id,
        display_name="Lab Virtual Organization",
        execution_mode=ExecutionMode.STRICT,
        channel_policy=ChannelPolicy(
            allowed_channels=["chat", "ticket", "email"],
            denied_channels=["phone", "sms"],
            default_channel="chat",
        ),
        tool_policy_overlay=ToolPolicyOverlay(
            deny_patterns=["phone.*", "sms.*"],
        ),
        communication_rules=CommunicationRules(
            required_disclosures=["org.disclosure.lab"],
            tone="formal",
        ),
        scenario_bindings=[
            ScenarioBinding(
                scenario_id="customer_intake",
                trigger="customer_service.intake",
                required_playbook_id="sop.customer_intake",
            ),
        ],
        compliance_profile_id="lab.org.compliance.v1",
        observability_labels={"org": organization_id, "sector": "lab"},
    )
