# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.steps.legal_extract_clauses_step import LegalExtractClausesStep
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer



class LegalAnalysisPipeline(RuntimePipeline):

    def __init__(self, *, config: LegalAgentConfig) -> None:
        super().__init__()
        self._config = config


    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        
        if state.agent_state is None:
            state.agent_state = LegalAgentState(
                config=self._config,
            )

        await LegalExtractClausesStep().run(state=state)

        return RuntimeAnswer(
            answer="OK"
        )