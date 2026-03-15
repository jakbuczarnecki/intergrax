# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

import pytest

from intergrax.rag.answers.answer_manager import AnswerManager
from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.engine.answer_engine import DefaultAnswerEngine

from intergrax.rag.answers.builders.context_builder import DefaultContextBuilder
from intergrax.rag.answers.builders.prompt_builder import DefaultPromptBuilder

from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)

from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.retrievers.engine.retriever_engine import RetrieverEngine
from intergrax.rag.retrievers.pipeline.retriever_pipeline import RetrieverPipeline
from intergrax.rag.retrievers.retriever_manager import RetrieverManager

from intergrax.tokenizers.bootstrap.tokenizer_bootstrap import create_default_tokenizer_manager

from tests._support.builder import FakeLLMAdapter


pytestmark = pytest.mark.unit


# ---------------------------------------------------------
# Fake embedding manager
# ---------------------------------------------------------

class _DummyEmbeddingManager:

    def embed_one(self, text: str):
        return [0.0]


# ---------------------------------------------------------
# Fake retriever
# ---------------------------------------------------------

class _EmptyRetriever(BaseRetriever):

    requires_query_embedding = False

    @classmethod
    def name(cls) -> str:
        return "empty"

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> List[RetrieverCandidate]:

        return []


class _DummyReranker:

    def rerank(self, *, query: str, candidates):
        return candidates


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

def test_rag_no_documents_returns_empty_context() -> None:

    retriever = _EmptyRetriever()

    registry = RetrieverRegistry([retriever])

    engine_retriever = RetrieverEngine(registry)

    pipeline = RetrieverPipeline(
        engine_retriever,
        embedding_manager=_DummyEmbeddingManager(),
    )

    retriever_manager = RetrieverManager(pipeline)

    tokenizer_manager = create_default_tokenizer_manager()

    engine = DefaultAnswerEngine(
        retriever_manager=retriever_manager,
        reranker_manager=_DummyReranker(),
        context_builder=DefaultContextBuilder(
            tokenizer_manager=tokenizer_manager
        ),
        prompt_builder=DefaultPromptBuilder(),
    )

    manager = AnswerManager(engine=engine)

    request = AnswerRequest(
        query="What is Intergrax?",
        llm=FakeLLMAdapter(),
    )

    result = manager.answer(request=request)

    assert result.answer
    assert result.context_documents == []
    assert result.pipeline_trace.retrieved_candidates == 0