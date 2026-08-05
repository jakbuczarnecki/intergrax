# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.rerankers.cache.rerank_cache import RerankCache
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerResult,
    RerankerCandidate,
)
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry


pytestmark = pytest.mark.unit


class FakeReranker(BaseReranker):

    def name(self) -> str:
        return "fake"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self,
        *,
        query: str | None,
        candidates: list[RerankerCandidate],
        limit: int | None = None,
    ) -> list[RerankerResult]:

        self.calls += 1

        results = []

        for i, candidate in enumerate(candidates, start=1):

            results.append(
                RerankerResult(
                    candidate=candidate,
                    rerank_score=float(len(candidate.document.content)),
                    fusion_score=None,
                    rank=i,
                )
            )

        if limit is not None:
            return results[:limit]

        return results


def _candidates() -> list[RerankerCandidate]:

    return [
        RerankerCandidate(
            document=KnowledgeDocument(
                schema_version=1,
                identity={"document_id": "1", "root_document_id": "1"},
                scope={"tenant_id": "unit", "namespace": "cache"},
                content="alpha",
                provenance={"source_kind": "test", "source_id": "1"},
            ),
            original_score=0.0,
            original_rank=0,
            channel="unit",
            vector_id="v1",
        ),
        RerankerCandidate(
            document=KnowledgeDocument(
                schema_version=1,
                identity={"document_id": "2", "root_document_id": "2"},
                scope={"tenant_id": "unit", "namespace": "cache"},
                content="beta",
                provenance={"source_kind": "test", "source_id": "2"},
            ),
            original_score=0.0,
            original_rank=1,
            channel="unit",
            vector_id="v2",
        ),
    ]


def test_cache_hit_skips_reranker() -> None:

    cache = RerankCache(ttl_seconds=60)

    registry = RerankerRegistry()

    reranker = FakeReranker()

    registry.register(reranker)

    engine = RerankerEngine(
        registry=registry,
        cache=cache,
    )

    candidates = _candidates()

    # first run -> cache miss
    engine.rerank(
        reranker_name="fake",
        query="query",
        candidates=candidates,
    )

    assert reranker.calls == 1

    # second run -> cache hit
    engine.rerank(
        reranker_name="fake",
        query="query",
        candidates=candidates,
    )

    assert reranker.calls == 1


def test_cache_miss_calls_reranker() -> None:

    cache = RerankCache(ttl_seconds=60)

    registry = RerankerRegistry()

    reranker = FakeReranker()

    registry.register(reranker)

    engine = RerankerEngine(
        registry=registry,
        cache=cache,
    )

    engine.rerank(
        reranker_name="fake",
        query="query",
        candidates=_candidates(),
    )

    assert reranker.calls == 1


def test_cache_store_after_miss() -> None:

    cache = RerankCache(ttl_seconds=60)

    registry = RerankerRegistry()

    reranker = FakeReranker()

    registry.register(reranker)

    engine = RerankerEngine(
        registry=registry,
        cache=cache,
    )

    candidates = _candidates()

    engine.rerank(
        reranker_name="fake",
        query="query",
        candidates=candidates,
    )

    result = cache.get(
        reranker="fake",
        query="query",
        texts=[c.document.content for c in candidates],
    )

    assert result is not None


def test_limit_on_cache_hit() -> None:

    cache = RerankCache(ttl_seconds=60)

    registry = RerankerRegistry()

    reranker = FakeReranker()

    registry.register(reranker)

    engine = RerankerEngine(
        registry=registry,
        cache=cache,
    )

    candidates = _candidates()

    engine.rerank(
        reranker_name="fake",
        query="query",
        candidates=candidates,
    )

    results = engine.rerank(
        reranker_name="fake",
        query="query",
        candidates=candidates,
        limit=1,
    )

    assert len(results) == 1