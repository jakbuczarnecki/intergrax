# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List, Optional, Sequence

from langchain_core.documents import Document
import pytest

from intergrax.rag.answers.contracts.base_context_builder import BaseContextBuilder
from intergrax.rag.answers.contracts.base_prompt_builder import BasePromptBuilder
from intergrax.rag.answers.engine.answer_engine import DefaultAnswerEngine
from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.contracts.answer_result import AnswerResult
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import Candidates, RerankerResult
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverCandidate
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from testing_support.builder import FakeLLMAdapter


pytestmark = pytest.mark.unit


class DummyRetrieverManager(BaseRetrieverManager):
    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> List[RetrieverCandidate]:
        return [
            RetrieverCandidate(
                id="id_0",
                content="Intergrax is an AI agent framework.",
                metadata={},
                score=0.9
            )
        ]


class DummyReranker(BaseReranker):
    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:
        return [
            RerankerResult(
                candidate=candidate,
                rerank_score=candidate.original_score or 0.0,
                fusion_score=None,
                rank=index,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
    
    @classmethod    
    def name(self) -> str:
        return "DummyReranker"

class DummyContextBuilder(BaseContextBuilder):
    def build(
        self,
        documents: List[Document],
    ) -> str:
        return "context"


class DummyPromptBuilder(BasePromptBuilder):
    def build(
        self,
        *,
        query: str,
        context: str,
    ) -> str:
        return f"{query}\n{context}"


def test_answer_engine_returns_answer_result_without_documents():
    engine = DefaultAnswerEngine(
        retriever_manager=DummyRetrieverManager(),
        reranker_manager=DummyReranker(),
        context_builder=DummyContextBuilder(),
        prompt_builder=DummyPromptBuilder(),        
    )

    request = AnswerRequest(
        query="What is Intergrax?",        
        llm=FakeLLMAdapter(),
        retriever_id="test"
    )

    result = engine.answer(request=request)

    assert isinstance(result, AnswerResult)
    assert result.answer is not None