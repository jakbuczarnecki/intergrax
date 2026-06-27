# © Artur Czarnecki. All rights reserved.

"""Organizational policy envelope — virtual workforce contract (architecture §39 · ACP-ORG-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.contracts.host_profile_slices import GuardrailProfile, PolicyRulesProfile


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
    """One simulated organization / tenant policy surface (ACP-ORG-1)."""

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
