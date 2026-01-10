# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.nexus.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.rag_step import RagStep
from intergrax.runtime.nexus.runtime_steps.retrieve_attachments_step import RetrieveAttachmentsStep
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.nexus.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from intergrax.runtime.nexus.runtime_steps.websearch_step import WebsearchStep


class NoPlannerPipeline(RuntimePipeline):

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        steps = [
            *SETUP_STEPS,

            RagStep(),
            UserLongtermMemoryStep(),
            RetrieveAttachmentsStep(),
            WebsearchStep(),

            ToolsStep(),
            CoreLLMStep(),
            PersistAndBuildAnswerStep(),
        ]

        await RuntimeStepRunner.execute_pipeline(steps, state)

        runtime_answer = state.runtime_answer
        if runtime_answer is None:
            raise RuntimeError("Persist step did not set state.runtime_answer.")
        
        return runtime_answer
    