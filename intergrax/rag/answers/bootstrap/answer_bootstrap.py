# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.answers.engine.answer_engine import DefaultAnswerEngine
from intergrax.rag.answers.builders.context_builder import DefaultContextBuilder
from intergrax.rag.answers.builders.prompt_builder import DefaultPromptBuilder
from intergrax.rag.answers.pipeline.answer_pipeline import AnswerPipeline
from intergrax.rag.retrievers.retriever_manager import RetrieverManager

from typing import Optional


from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)

from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import (
    create_default_reranker_engine,
)

from intergrax.rag.retrievers.retriever_manager import RetrieverManager
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine


def create_default_answer_pipeline(
    *,
    retriever_manager: Optional[RetrieverManager] = None,
    reranker_engine: Optional[RerankerEngine] = None,
) -> AnswerPipeline:

    if retriever_manager is None:
        retriever_manager = create_default_retriever_manager()

    if reranker_engine is None:
        reranker_engine = create_default_reranker_engine()

    context_builder = DefaultContextBuilder()
    prompt_builder = DefaultPromptBuilder()

    return AnswerPipeline(
        retriever_manager=retriever_manager,
        reranker_manager=reranker_engine,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )


def create_default_answer_engine(
    *,
    llm: Optional[LLMAdapter] = None,
    retriever_manager: Optional[RetrieverManager] = None,
    reranker_engine: Optional[RerankerEngine] = None,
) -> DefaultAnswerEngine:

    pipeline = create_default_answer_pipeline(
        retriever_manager=retriever_manager,
        reranker_engine=reranker_engine,
    )

    return DefaultAnswerEngine(
        llm=llm,
        pipeline=pipeline,
    )