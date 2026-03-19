# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.session.session_manager import SessionManager

from intergrax.agents_packages.legal_agent.pipeline import LegalAnalysisPipeline


class LegalAgent(Agent):
    """
    Real business agent: contract analysis.
    """

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        llm_adapter: LLMAdapter,
        production_mode: bool = True,
    ) -> None:
        self._session_manager = session_manager
        self._llm_adapter = llm_adapter
        self._production_mode = production_mode

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter = self._llm_adapter,
            enable_rag=False,
            production_mode=self._production_mode
        )

        # --- PIPELINE (CRITICAL) ---
        config.pipeline = LegalAnalysisPipeline()

        # --- BUILD CONTEXT ---
        context = RuntimeContext.build(
            config=config,
            session_manager=self._session_manager,
        )

        return context