# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.legacy.rag_answers.contracts.answer_request import AnswerRequest
from intergrax.legacy.rag_answers.contracts.answer_result import AnswerResult


from intergrax.legacy.rag_answers.contracts.base_context_builder import BaseContextBuilder
from intergrax.legacy.rag_answers.contracts.base_prompt_builder import BasePromptBuilder
from intergrax.legacy.rag_answers.pipeline.pipeline_trace import PipelineTrace, StepTimer
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager

from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager


class AnswerPipeline:

    def __init__(
        self,
        *,
        retriever_manager: BaseRetrieverManager,
        reranker_manager: BaseRerankerManager,
        context_builder: BaseContextBuilder,
        prompt_builder: BasePromptBuilder,
    ) -> None:

        self._retriever_manager = retriever_manager
        self._reranker_manager = reranker_manager
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder

    def run(
        self,
        *,
        request: AnswerRequest,
    ) -> AnswerResult:
        
        trace = PipelineTrace()
        
        # STEP 1 — retrieval
        timer = StepTimer()
        retrieved_candidates = self._retriever_manager.retrieve(
            query_text=request.query,
            retriever_id=request.retriever_id,
            query_embedding=None,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
            include_embeddings=request.include_embeddings
        )

        trace.retrieval_latency_ms = timer.stop_ms()
        trace.retrieved_candidates = len(retrieved_candidates)

        # STEP 2 — convert to reranker candidates
        reranker_candidates = [
            RerankerCandidate(
                id=c.id,
                text=c.content,
                metadata=c.metadata,
                original_score=c.score,
            )
            for c in retrieved_candidates
        ]

        # STEP 3 — rerank
        timer = StepTimer()
        rerank_results = self._reranker_manager.rerank(
            query=request.query,
            candidates=reranker_candidates,
        )
        trace.rerank_latency_ms = timer.stop_ms()
        trace.reranked_candidates = len(rerank_results)

        # STEP 4 — extract documents
        reranked_docs: List[Document] = [
            Document(
                page_content=result.candidate.text,
                metadata=result.candidate.metadata,
            )
            for result in rerank_results
        ]

        # STEP 5 — context building
        timer = StepTimer()
        context = self._context_builder.build(
            documents=reranked_docs
        )
        trace.context_latency_ms = timer.stop_ms()

        # STEP 6 — prompt
        timer = StepTimer()
        prompt = self._prompt_builder.build(
            query=request.query,
            context=context,
        )
        trace.prompt_latency_ms = timer.stop_ms()

        # STEP 7 — LLM call
        messages = [
            ChatMessage(
                role="user",
                content=prompt,
            )
        ]

        timer = StepTimer()
        answer_response = request.llm.generate_messages(messages)
        trace.llm_latency_ms = timer.stop_ms()

        result = AnswerResult(
            answer=answer_response.content,
            context_documents=reranked_docs,
        )

        result.pipeline_trace = trace

        return result