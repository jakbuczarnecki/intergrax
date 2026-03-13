# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.answers.contracts.answer_engine import AnswerEngine
from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.contracts.answer_result import AnswerResult

from intergrax.rag.answers.contracts.base_context_builder import BaseContextBuilder
from intergrax.rag.answers.contracts.base_prompt_builder import BasePromptBuilder
from intergrax.rag.answers.pipeline.answer_pipeline import AnswerPipeline
from intergrax.rag.rerankers.re_ranker import ReRanker
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager


class DefaultAnswerEngine(AnswerEngine):

    def __init__(
        self,
        *,
        retriever_manager: BaseRetrieverManager,
        reranker_manager: ReRanker,
        context_builder: BaseContextBuilder,
        prompt_builder: BasePromptBuilder,        
    ) -> None:
                
        self._retriever_manager = retriever_manager
        self._reranker_manager = reranker_manager
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder

        self._pipeline = AnswerPipeline(
            retriever_manager=self._retriever_manager,
            reranker_manager=self._reranker_manager,
            context_builder=self._context_builder,
            prompt_builder=self._prompt_builder,
        )


    def answer(
        self,
        *,
        request: AnswerRequest,
    ) -> AnswerResult:
        return self._pipeline.run(request=request)
