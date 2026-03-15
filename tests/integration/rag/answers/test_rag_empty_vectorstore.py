# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.rag.answers.answer_manager import AnswerManager
from intergrax.rag.answers.builders.context_builder import DefaultContextBuilder
from intergrax.rag.answers.builders.prompt_builder import DefaultPromptBuilder
from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.engine.answer_engine import DefaultAnswerEngine
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_manager,
)
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import (
    create_default_reranker_engine,
)
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import (
    create_default_vectorstore_manager,
)
from intergrax.tokenizers.bootstrap.tokenizer_bootstrap import create_default_tokenizer


pytestmark = pytest.mark.integration


def test_rag_pipeline_empty_vectorstore() -> None:
    """
    Integration test verifying pipeline behaviour when vectorstore is empty.
    """

    vectorstore_manager = create_default_vectorstore_manager()

    embedding_manager = create_default_embedding_manager()

    retriever_manager = create_default_retriever_manager(
        vector_store=vectorstore_manager,
        embedding_manager=embedding_manager,
    )

    reranker = ReRankerManager(
        engine=create_default_reranker_engine(
            embedding_manager=embedding_manager,
        )
    )

    answer_engine = DefaultAnswerEngine(
        retriever_manager=retriever_manager,
        reranker_manager=reranker,
        context_builder=DefaultContextBuilder(
            tokenizer_manager=create_default_tokenizer(),
        ),
        prompt_builder=DefaultPromptBuilder(),
    )

    manager = AnswerManager(engine=answer_engine)

    request = AnswerRequest(
        query="What is the capital of France?",
        top_k=3,
        llm=LangChainOllamaAdapter(),
    )

    result = manager.answer(request=request)

    assert result is not None
    assert result.answer
    assert isinstance(result.answer, str)

    # retrieval should return empty
    assert result.pipeline_trace.retrieved_candidates == 0

    # rerank should also be empty
    assert result.pipeline_trace.reranked_candidates == 0