# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod

from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStep


class LegalBaseStep(RuntimeStep, ABC):

    def execution_kind(self) -> ExecutionKind | None:
        return ExecutionKind.LLM

    async def run(self, state: RuntimeState) -> None:

        if state.agent_state is None:
            raise RuntimeError("agent_state is not initialized.")

        if not isinstance(state.agent_state, LegalAgentState):
            raise TypeError("state.agent_state must be LegalAgentState.")
        
        await self.run_step(state=state, agent_state=state.agent_state)
        

    @abstractmethod
    async def run_step(
        self,         
        state: RuntimeState,
        agent_state: LegalAgentState
    ) -> None:
        raise NotImplementedError()