# © Artur Czarnecki. All rights reserved.

"""Verified Agent Governance approval token (minted only by verifier module)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceHumanApprovalGrant,
    AgentGovernanceHumanApprovalRequirement,
    LogicalInvocationFingerprint,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)

SCHEMA_VERIFIED_AGENT_GOVERNANCE_HUMAN_APPROVAL_V1: Final = (
    "verified_agent_governance_human_approval.v1"
)

_NON_EMPTY = Field(min_length=1)


class VerifiedAgentGovernanceHumanApproval(BaseModel):
    """Immutable proof of successful AgentGovernanceGrantVerifier verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_VERIFIED_AGENT_GOVERNANCE_HUMAN_APPROVAL_V1
    verification_id: str = _NON_EMPTY
    grant_id: str = _NON_EMPTY
    agent_governance_invocation_scope_id: str = _NON_EMPTY
    requirement: AgentGovernanceHumanApprovalRequirement
    grant: AgentGovernanceHumanApprovalGrant
    logical_invocation_fingerprint: LogicalInvocationFingerprint
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    pause_generation: int = Field(ge=1)


__all__ = ["VerifiedAgentGovernanceHumanApproval"]
