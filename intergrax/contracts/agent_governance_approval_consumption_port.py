# © Artur Czarnecki. All rights reserved.

"""Typed boundary: fresh Agent Governance ALLOW → grant lifecycle APPLIED (UCA-6C-R6-R5.6-H1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.agent_governance_verified_approval import (
    VerifiedAgentGovernanceHumanApproval,
)


class AgentGovernanceApprovalConsumptionError(RuntimeError):
    """Fail-closed when RESERVED → APPLIED cannot be persisted after governance ALLOW."""


class AgentGovernanceApprovalConsumptionPort(ABC):
    """Marks human approval grant consumed exactly after successful fresh Agent Governance ALLOW."""

    @abstractmethod
    def mark_applied_after_governance_allow(
        self,
        verified: VerifiedAgentGovernanceHumanApproval,
    ) -> None:
        """Persist APPLIED via canonical grant lifecycle; raise on CAS / ownership failure."""


__all__ = [
    "AgentGovernanceApprovalConsumptionError",
    "AgentGovernanceApprovalConsumptionPort",
]
