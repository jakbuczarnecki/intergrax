# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.rag.rerankers.providers.semantic_reranker import SemanticReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerResult,
)
from tests.integration.rag.rerankers._datasets import candidates


pytestmark = pytest.mark.integration


def test_semantic_reranker_basic() -> None:

    reranker = SemanticReranker()

    results = reranker.rerank(
        query="What is the capital of France?",
        candidates=candidates(),
    )

    assert isinstance(results, list)

    assert all(isinstance(r, RerankerResult) for r in results)

    assert len(results) == 3

    assert results[0].rank == 1


def test_semantic_reranker_limit() -> None:

    reranker = SemanticReranker()

    results = reranker.rerank(
        query="What is the capital of France?",
        candidates=candidates(),
        limit=1,
    )

    assert len(results) == 1

    assert results[0].rank == 1