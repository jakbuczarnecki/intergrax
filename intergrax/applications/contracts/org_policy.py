# © Artur Czarnecki. All rights reserved.

"""Organizational policy envelope — compatibility re-export and product fixtures."""

from __future__ import annotations

from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.contracts.org_policy import (
    ChannelPolicy,
    CommunicationRules,
    OrganizationalPolicyContext,
    OrganizationalPolicyEnvelope,
    ScenarioBinding,
    ToolPolicyOverlay,
)

__all__ = [
    "ChannelPolicy",
    "CommunicationRules",
    "ExecutionMode",
    "OrganizationalPolicyContext",
    "OrganizationalPolicyEnvelope",
    "ScenarioBinding",
    "ToolPolicyOverlay",
    "lab_strict_org_envelope",
    "product_host_org_envelope",
]


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
