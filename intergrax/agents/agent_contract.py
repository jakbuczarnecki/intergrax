# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_run_enums import AgentRunStatus, TerminalReason
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope, routing_capability_from_envelope
from intergrax.contracts.validation import ValidationResult


class Agent(ABC):
    """
    Tier-2 Agent contract.

    Public I/O is ``AgentRunRequest`` → ``AgentRunResult``.
    Nexus ``RuntimeRequest`` / ``RuntimeContext`` materialization stays in runtime composition.
    """

    @abstractmethod
    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Execute this agent for the typed public run contract."""
        ...

    def get_contract(self) -> AgentContract:
        """Return declarative agent metadata. Override in concrete agents."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_contract() "
            "or register metadata via AgentRegistry."
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        """Optional capability pre-check. Default: no match."""
        _ = routing_capability_from_envelope(task)
        return CapabilityMatchResult(
            matched=False,
            rationale=f"{type(self).__name__} does not implement can_handle()",
        )

    def validate(self, result: AgentRunResult) -> ValidationResult:
        """Optional local output validation. Default: pass if output non-empty."""
        if result.status != AgentRunStatus.SUCCEEDED:
            return ValidationResult(
                valid=False,
                errors=[error.message for error in result.errors] or ["run failed"],
            )
        if isinstance(result.output, str):
            if result.output.strip():
                return ValidationResult(valid=True)
            return ValidationResult(valid=False, errors=["empty output"])
        if result.output:
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, errors=["empty output"])
