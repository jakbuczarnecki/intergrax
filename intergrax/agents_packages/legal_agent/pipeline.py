# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List

from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.llm.messages import ChatMessage


class LegalAnalysisPipeline(RuntimePipeline):
    """
    Legal analysis pipeline (v1).

    - single LLM call
    - uses LLMAdapter.generate_messages(...)
    - builds ChatMessage list explicitly
    """

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        cfg = state.context.config
        llm = cfg.llm_adapter

        if llm is None:
            raise RuntimeError("LegalAnalysisPipeline: llm_adapter is not configured.")

        contract_text = state.request.message or ""

        system_prompt = (
            "You are a senior legal expert. "
            "Analyze contracts and provide structured, clear output."
        )

        user_prompt = (
            "Analyze the contract below and provide:\n\n"
            "1. SUMMARY - short overview of the contract\n"
            "2. KEY CLAUSES -list of the most important clauses\n"
            "3. RISKS - list of potential risks with explanation\n\n"
            "Contract:\n"
            f"{contract_text}"
        )

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        response_text = llm.generate_messages(
            messages=messages,
            run_id=state.run_id,
        )

        answer = RuntimeAnswer(
            run_id=state.run_id,
            answer=response_text,
        )

        state.runtime_answer = answer
        state.raw_answer = response_text

        return state.runtime_answer