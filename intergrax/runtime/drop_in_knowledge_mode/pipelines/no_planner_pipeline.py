# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.rag_step import RagStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.retrieve_attachments_step import RetrieveAttachmentsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.websearch_step import WebsearchStep


class NoPlannerPipeline(RuntimePipeline):

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        steps = [
            *SETUP_STEPS,

            # Must exist before any step that injects context "before last user".
            EnsureCurrentUserMessageStep(),

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
    