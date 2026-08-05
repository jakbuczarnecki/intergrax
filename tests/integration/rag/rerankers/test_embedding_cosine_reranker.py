# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_pipeline
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.rerankers.providers.embedding_cosine_reranker import (
    EmbeddingCosineReranker,
)
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerResult,
)
from tests.integration.rag.rerankers._datasets import candidates


pytestmark = pytest.mark.integration



def test_embedding_cosine_reranker_basic() -> None:

    embed_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(),
    )

    reranker = EmbeddingCosineReranker(embed_manager)

    results = reranker.rerank(
        query="What is the capital of France?",
        candidates=candidates(),
    )

    assert isinstance(results, list)

    assert all(isinstance(r, RerankerResult) for r in results)

    assert len(results) == 3

    assert results[0].rank == 0


def test_embedding_cosine_reranker_limit() -> None:

    embed_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(),
    )

    reranker = EmbeddingCosineReranker(embed_manager)

    results = reranker.rerank(
        query="What is the capital of France?",
        candidates=candidates(),
        limit=1,
    )

    assert len(results) == 1

    assert results[0].rank == 0