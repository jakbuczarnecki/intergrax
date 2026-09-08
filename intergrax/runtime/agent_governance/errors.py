# © Artur Czarnecki. All rights reserved.

"""Governance boundary errors at the tool invocation layer."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_runtime_governance import (
    PolicyEvaluationResult,
    ToolAuthorizationDecisionState,
)


@dataclass(frozen=True)
class ToolGovernanceDeniedError(RuntimeError):
    """Raised when agent runtime governance denies a tool invocation."""

    run_id: str
    agent_id: str
    tool_id: str
    capability: str
    reason: str
    policy_results: tuple[PolicyEvaluationResult, ...]

    def __str__(self) -> str:
        return (
            f"Agent runtime governance denied tool '{self.tool_id}' "
            f"(agent='{self.agent_id}', capability='{self.capability}', "
            f"run_id={self.run_id}): {self.reason}"
        )


@dataclass(frozen=True)
class ToolGovernanceApprovalRequiredError(RuntimeError):
    """Typed boundary signal: governance requires human approval before tool execution."""

    run_id: str
    agent_id: str
    tool_id: str
    capability: str
    approval_id: str
    reason: str
    policy_results: tuple[PolicyEvaluationResult, ...]

    def __str__(self) -> str:
        return (
            f"Agent runtime governance requires human approval for tool '{self.tool_id}' "
            f"(agent='{self.agent_id}', approval_id={self.approval_id}, "
            f"run_id={self.run_id}): {self.reason}"
        )


@dataclass(frozen=True)
class CapabilityNotGrantedError(RuntimeError):
    """Raised when agent lacks capability grant for requested action."""

    run_id: str
    agent_id: str
    capability: str
    tool_id: str

    def __str__(self) -> str:
        return (
            f"Capability '{self.capability}' not granted for agent '{self.agent_id}' "
            f"(tool='{self.tool_id}', run_id={self.run_id})."
        )
