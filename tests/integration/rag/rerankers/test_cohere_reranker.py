# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
import pytest

from intergrax.rag.rerankers.providers.cohere_reranker import CohereReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerResult,
)
from tests.integration.rag.rerankers._datasets import candidates


pytestmark = pytest.mark.integration


def _skip_if_missing_key() -> None:

    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set")


def test_cohere_reranker_basic() -> None:

    _skip_if_missing_key()

    reranker = CohereReranker()

    results = reranker.rerank(
        query="What is the capital of France?",
        candidates=candidates(),
    )

    assert isinstance(results, list)

    assert all(isinstance(r, RerankerResult) for r in results)

    assert len(results) == 3

    assert results[0].rank == 0


def test_cohere_reranker_limit() -> None:

    _skip_if_missing_key()

    reranker = CohereReranker()

    results = reranker.rerank(
        query="What is the capital of France?",
        candidates=candidates(),
        limit=1,
    )

    assert len(results) == 1

    assert results[0].rank == 0