# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Legal Agent pipeline with:
  - Nexus SETUP_STEPS (session load, user message persistence, base history — conversational memory)
  - LLM-assisted routing of analysis stages (see :mod:`~legal.pipeline.legal_pipeline_routing`)
  - Final answer step always runs
"""

from __future__ import annotations

from legal.config.legal_agent_config import LegalAgentConfig
from legal.domain.legal_agent_state import LegalAgentState
from legal.memory.legal_memory_policy import (
    resolve_session_prior_workspace_snapshot,
)
from legal.pipeline.legal_execution_loop import (
    run_legal_dynamic_execution_loop,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS


class LegalDynamicPipeline(RuntimePipeline):

    def __init__(self, *, config: LegalAgentConfig) -> None:
        super().__init__()
        self._config = config

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await RuntimeStepRunner.execute_pipeline(SETUP_STEPS, state)

        policy = self._config.memory_policy
        prior_snapshot = resolve_session_prior_workspace_snapshot(
            session=state.session,
            request=state.request,
            policy=policy,
        )

        if state.agent_state is None:
            state.agent_state = LegalAgentState(
                config=self._config,
                session_prior_workspace_snapshot=prior_snapshot,
            )
        elif not isinstance(state.agent_state, LegalAgentState):
            raise TypeError("state.agent_state must be LegalAgentState for LegalDynamicPipeline.")
        else:
            state.agent_state = state.agent_state.model_copy(
                update={"session_prior_workspace_snapshot": prior_snapshot}
            )

        await run_legal_dynamic_execution_loop(
            state=state,
            agent_state=state.agent_state,
            config=self._config,
        )

        if state.runtime_answer is None:
            raise RuntimeError("Legal pipeline did not set state.runtime_answer.")

        return state.runtime_answer
