# © Artur Czarnecki. All rights reserved.

"""Agent lifecycle governance contracts for Tier-3 hosts (APP-EVOL-4 · §49.4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState


class AgentApprovalPolicy(BaseModel):
    """Lifecycle states permitted on STRICT product rosters (§49.4.2)."""

    model_config = ConfigDict(extra="forbid")

    allowed_states_for_strict: list[AgentLifecycleState] = Field(
        default_factory=lambda: [
            AgentLifecycleState.PRODUCTION,
            AgentLifecycleState.STAGING,
        ],
    )
    allow_staging_in_balanced: bool = True


class AgentCertificationRecord(BaseModel):
    """Certification evidence for a roster agent on a Tier-3 host (§49.4.2)."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    certified_at: str = Field(min_length=1, description="UTC ISO-8601 timestamp")
    certified_by: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_refs_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("evidence_refs must include at least one non-empty reference")
        return cleaned


class AgentGovernanceProfile(BaseModel):
    """Environment-scoped agent approval and certification registry."""

    model_config = ConfigDict(extra="forbid")

    approval_policy: AgentApprovalPolicy = Field(default_factory=AgentApprovalPolicy)
    certifications: list[AgentCertificationRecord] = Field(default_factory=list)
