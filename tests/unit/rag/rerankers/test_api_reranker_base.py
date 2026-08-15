# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.rerank_provider.cohere_rerank import opens as cohere_opens
from intergrax.integrations.providers.rerank_provider.jina_rerank import opens as jina_opens
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit

from intergrax.rag.rerankers.providers._api_reranker_base import _APIRerankerBase
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
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


def build_candidates(workspace_id: str | None = None):

    return [
        RerankerCandidate(
            document=KnowledgeDocument(
                schema_version=1,
                identity={"document_id": key, "root_document_id": key},
                scope={
                    "tenant_id": "unit",
                    "namespace": "rerank",
                    "workspace_id": workspace_id,
                },
                content=f"doc {key}",
                provenance={"source_kind": "test", "source_id": key},
            ),
            original_score=0.5,
            original_rank=rank,
            channel="unit",
            vector_id=f"v-{key}",
        )
        for rank, key in enumerate(("a", "b", "c"))
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

    assert [result.rank for result in results] == [0, 1, 2]


def test_rerank_respects_limit():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="test",
        candidates=build_candidates(),
        limit=2,
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [0, 1]


def test_empty_candidates_returns_empty():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="test",
        candidates=[],
    )

    assert results == ()


def test_empty_query_returns_empty():

    reranker = FakeAPIReranker()

    results = reranker.rerank(
        query="",
        candidates=build_candidates(),
    )

    assert results == ()


def test_candidate_is_native_immutable_and_defensively_revalidated():
    source = build_candidates()[0]
    candidate = RerankerCandidate(
        document=source.document,
        original_score=0.7,
        original_rank=4,
        channel="vector",
        vector_id="vector-4",
    )

    assert candidate.document == source.document
    assert candidate.document is not source.document
    with pytest.raises(FrozenInstanceError):
        candidate.original_rank = 5
    with pytest.raises(TypeError):
        RerankerCandidate(
            document=source.document,
            original_score=True,
            original_rank=0,
            channel="vector",
        )
    with pytest.raises(ValueError):
        RerankerCandidate(
            document=source.document,
            original_score=math.nan,
            original_rank=0,
            channel="vector",
        )
    with pytest.raises(TypeError):
        RerankerCandidate(
            document=source.document,
            original_score=0.5,
            original_rank=True,
            channel="vector",
        )
    with pytest.raises(ValueError):
        RerankerCandidate(
            document=source.document,
            original_score=0.5,
            original_rank=0,
            channel=" ",
        )
    with pytest.raises(ValueError):
        RerankerCandidate(
            document=source.document,
            original_score=0.5,
            original_rank=0,
            channel="vector",
            vector_id=" ",
        )


def test_candidate_from_retrieval_hit_preserves_native_fields():
    source = build_candidates()[0]
    hit = RetrievalHit(
        document=source.document,
        score=0.8,
        rank=3,
        channel="hybrid",
        vector_id="vector-3",
    )

    candidate = RerankerCandidate.from_retrieval_hit(hit)

    assert candidate.document.identity == hit.document.identity
    assert candidate.document.scope == hit.document.scope
    assert candidate.document.provenance == hit.document.provenance
    assert candidate.document.metadata == hit.document.metadata
    assert candidate.original_score == hit.score
    assert candidate.original_rank == hit.rank
    assert candidate.channel == hit.channel
    assert candidate.vector_id == hit.vector_id


def test_candidate_identity_key_includes_workspace() -> None:
    candidate_a = build_candidates("workspace-a")[0]
    candidate_b = build_candidates("workspace-b")[0]

    assert candidate_a.identity_key != candidate_b.identity_key


def test_result_is_immutable_finite_and_serializable():
    candidate = build_candidates()[0]
    result = RerankerResult(
        candidate=candidate,
        rerank_score=0.9,
        fusion_score=0.8,
        rank=0,
    )

    assert result.to_json() == result.to_json()
    with pytest.raises(FrozenInstanceError):
        result.rank = 1
    with pytest.raises(ValueError):
        RerankerResult(candidate=candidate, rerank_score=math.inf, rank=0)
    with pytest.raises(ValueError):
        RerankerResult(candidate=candidate, rerank_score=0.9, fusion_score=math.nan, rank=0)
    with pytest.raises(TypeError):
        RerankerResult(candidate=candidate, rerank_score=0.9, rank=True)


def test_cohere_provider_maps_indices_and_skips_empty_batch(monkeypatch):
    calls = []

    class FakeClient:
        def rerank(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.9),
                    SimpleNamespace(index=0, relevance_score=0.2),
                ]
            )

    monkeypatch.setattr(cohere_opens, "_client", lambda _config: FakeClient())
    config = SimpleNamespace(model="test", top_n=None)

    assert cohere_opens.cohere_rerank_scores(config, "query", ["a", "b"]) == [0.2, 0.9]
    assert cohere_opens.cohere_rerank_scores(config, "query", []) == []
    assert len(calls) == 1


@pytest.mark.parametrize(
    "results",
    [
        [SimpleNamespace(index=2, relevance_score=0.5)],
        [
            SimpleNamespace(index=0, relevance_score=0.5),
            SimpleNamespace(index=0, relevance_score=0.4),
        ],
        [SimpleNamespace(index=0, relevance_score=math.nan)],
    ],
)
def test_cohere_provider_rejects_invalid_response(monkeypatch, results):
    monkeypatch.setattr(
        cohere_opens,
        "_client",
        lambda _config: SimpleNamespace(
            rerank=lambda **_kwargs: SimpleNamespace(results=results)
        ),
    )
    with pytest.raises((TypeError, ValueError)):
        cohere_opens.cohere_rerank_scores(
            SimpleNamespace(model="test", top_n=None),
            "query",
            ["a", "b"],
        )


def test_jina_provider_rejects_invalid_index(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 4, "relevance_score": 0.5}]}

    monkeypatch.setattr(jina_opens.requests, "post", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(ValueError):
        jina_opens.jina_rerank_scores(
            SimpleNamespace(api_key="test", model="test", api_url="https://example.test"),
            "query",
            ["a"],
        )