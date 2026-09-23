# © Artur Czarnecki. All rights reserved.

"""Typed policy evaluation context for Agent Runtime Governance (UCA-6C-R6-R5)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.agent_governance_verified_approval import (
    VerifiedAgentGovernanceHumanApproval,
)


class AgentRuntimePolicyEvaluationContext(BaseModel):
    """Context passed to policy providers — no store/verifier dependencies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_agent_governance_human_approval: VerifiedAgentGovernanceHumanApproval | None = (
        None
    )


__all__ = ["AgentRuntimePolicyEvaluationContext"]
