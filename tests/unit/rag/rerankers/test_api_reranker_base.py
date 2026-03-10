# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.rag.rerankers.providers._api_reranker_base import _APIRerankerBase
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
)


pytestmark = pytest.mark.unit


class FakeAPIReranker(_APIRerankerBase):

    @classmethod
    def name(cls) -> str:
        return "fake_api"

    def _score(self, query, texts):

        # deterministic scoring for testing
        scores = []

        for i, _ in enumerate(texts):
            scores.append(float(i))

        return scores


def build_candidates():

    return [
        RerankerCandidate(id="a", text="doc a", metadata={}, original_score=None),
        RerankerCandidate(id="b", text="doc b", metadata={}, original_score=None),
        RerankerCandidate(id="c", text="doc c", metadata={}, original_score=None),
    ]


def test_rerank_returns_sorted_results():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="test",
        candidates=build_candidates(),
    )

    assert len(results) == 3

    assert results[0].rerank_score >= results[1].rerank_score
    assert results[1].rerank_score >= results[2].rerank_score


def test_rerank_assigns_rank():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="test",
        candidates=build_candidates(),
    )

    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3


def test_rerank_respects_limit():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="test",
        candidates=build_candidates(),
        limit=2,
    )

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_empty_candidates_returns_empty():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="test",
        candidates=[],
    )

    assert results == []


def test_empty_query_returns_empty():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="",
        candidates=build_candidates(),
    )

    assert results == []