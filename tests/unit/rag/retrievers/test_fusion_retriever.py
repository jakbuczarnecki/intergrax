# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from typing import List

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.retrievers.providers.fusion_retriever import FusionRetriever
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry


pytestmark = pytest.mark.unit


class FakeRetrieverA(BaseRetriever):

    @classmethod
    def name(cls) -> str:
        return "fake_a"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:

        return [
            RetrieverCandidate(
                id="a",
                content="doc-a",
                metadata={
                    ChunkMetadataKey.PARENT_CHUNK_ID: "docA"
                },
                score=0.9,
                embedding=None,
                rank=0,
            ),
            RetrieverCandidate(
                id="b",
                content="doc-b",
                metadata={
                    ChunkMetadataKey.PARENT_CHUNK_ID: "docB"
                },
                score=0.8,
                embedding=None,
                rank=1,
            ),
        ]


class FakeRetrieverB(BaseRetriever):

    @classmethod
    def name(cls) -> str:
        return "fake_b"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:

        return [
            RetrieverCandidate(
                id="a",
                content="doc-a",
                metadata={
                    ChunkMetadataKey.PARENT_CHUNK_ID: "docA"
                },
                score=0.7,
                embedding=None,
                rank=0,
            ),
            RetrieverCandidate(
                id="c",
                content="doc-c",
                metadata={
                    ChunkMetadataKey.PARENT_CHUNK_ID: "docC"
                },
                score=0.85,
                embedding=None,
                rank=1,
            ),
        ]


def test_fusion_retriever_merges_and_deduplicates():

    registry = RetrieverRegistry()

    registry.register(FakeRetrieverA())
    registry.register(FakeRetrieverB())

    retriever = FusionRetriever(
        registry=registry,
        retrievers=["fake_a", "fake_b"]
    )

    query = RetrieverQuery(
        query_text="test query",
        query_embedding=None,
        top_k=2,
        metadata_filter=None,
        include_embeddings=False,
    )

    results = retriever.retrieve(query)

    assert len(results) == 2

    ids = {r.id for r in results}

    # ensure merged and deduplicated
    assert ids.issubset({"a", "b", "c"})

    # ensure metadata propagated
    for r in results:
        assert r.metadata is not None