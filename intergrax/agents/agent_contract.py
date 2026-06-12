# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.task.task import TaskContext


class Agent(ABC):
    """
    Tier-2 Agent contract.

    Agent is responsible for:
    - building RuntimeContext (including RuntimeConfig)
    - declaring capabilities via get_contract()
    - cognitive behavior via ACP (``on_next_step``) or UAEP steps

    Agent is NOT responsible for:
    - RuntimeState
    - execution lifecycle
    - global orchestration
    """

    @abstractmethod
    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        """
        Build fully configured RuntimeContext for this agent.
        Must include RuntimeConfig and dependencies required for this agent's run.
        """
        ...

    def get_contract(self) -> AgentContract:
        """Return declarative agent metadata. Override in concrete agents."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_contract() "
            "or register metadata via AgentRegistry."
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        """Optional capability pre-check. Default: no match."""
        return CapabilityMatchResult(
            matched=False,
            rationale=f"{type(self).__name__} does not implement can_handle()",
        )

    def validate(
        self,
        output: RuntimeAnswer,
        *,
        context: Optional[RuntimeContext] = None,
    ) -> ValidationResult:
        """Optional local output validation. Default: pass if answer non-empty."""
        if output.answer and output.answer.strip():
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, errors=["empty answer"])

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Execute this agent for the given request.

        Delegates to :meth:`intergrax.agents.agent_engine.AgentEngine.run_agent`
        so all Tier-2 agents share one runtime path.
        """
        from intergrax.agents.agent_engine import AgentEngine

        return await AgentEngine.run_agent(self, request)
