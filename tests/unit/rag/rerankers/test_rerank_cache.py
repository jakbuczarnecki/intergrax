# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time

import pytest

from intergrax.rag.rerankers.cache.rerank_cache import RerankCache


pytestmark = pytest.mark.unit


def test_cache_miss_returns_none() -> None:

    cache = RerankCache(ttl_seconds=60)

    result = cache.get(
        reranker="test",
        query="query",
        texts=["doc1", "doc2"],
    )

    assert result is None


def test_cache_set_and_get() -> None:

    cache = RerankCache(ttl_seconds=60)

    scores = [0.9, 0.7]

    cache.set(
        reranker="test",
        query="query",
        texts=["doc1", "doc2"],
        scores=scores,
    )

    result = cache.get(
        reranker="test",
        query="query",
        texts=["doc1", "doc2"],
    )

    assert result == scores


def test_cache_normalization_whitespace() -> None:

    cache = RerankCache(ttl_seconds=60)

    scores = [0.8]

    cache.set(
        reranker="test",
        query="query",
        texts=["doc"],
        scores=scores,
    )

    result = cache.get(
        reranker="test",
        query="query",
        texts=["  doc  "],
    )

    assert result == scores


def test_cache_ttl_expiration(monkeypatch: pytest.MonkeyPatch) -> None:

    cache = RerankCache(ttl_seconds=1)
    now = 1_000.0
    monkeypatch.setattr(time, "time", lambda: now)

    cache.set(
        reranker="test",
        query="query",
        texts=["doc"],
        scores=[0.5],
    )

    now += 1.2

    result = cache.get(
        reranker="test",
        query="query",
        texts=["doc"],
    )

    assert result is None


def test_hash_determinism() -> None:

    cache = RerankCache(ttl_seconds=60)

    texts1 = ["doc1", "doc2"]
    texts2 = ["doc1", "doc2"]

    h1 = cache._hash_documents(texts1)
    h2 = cache._hash_documents(texts2)

    assert h1 == h2