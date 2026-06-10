# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.governance.embedding_version_policy import (
    ReindexQueueRequest,
    clear_reindex_queue_hooks,
    evaluate_ingest_embedding_version,
    filter_chunks_by_embedding_version,
    register_reindex_queue_hook,
)
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _clear_hooks() -> None:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


def test_evaluate_ingest_warns_on_incoming_metadata_mismatch() -> None:
    profile = RagProfile(embedding_model_version="text-embedding-3-large")
    result = evaluate_ingest_embedding_version(
        profile=profile,
        base_metadata={"embedding_model_version": "text-embedding-ada-002"},
        source_path="/tmp/doc.pdf",
    )
    assert any("incoming_metadata_version_mismatch" in warning for warning in result.warnings)


def test_reindex_hook_invoked_on_mismatch() -> None:
    queued: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(queued.append)

    profile = RagProfile(embedding_model_version="v2")
    result = evaluate_ingest_embedding_version(
        profile=profile,
        base_metadata={"embedding_model_version": "v1"},
        source_path="/data/corpus.pdf",
    )

    assert result.reindex_enqueued is True
    assert len(queued) == 1
    assert queued[0].source_path == "/data/corpus.pdf"
    assert queued[0].current_version == "v1"
    assert queued[0].target_version == "v2"


def test_filter_chunks_by_embedding_version_drops_mismatched_only() -> None:
    profile = RagProfile(
        embedding_model_version="v2",
        embedding_version_filter_on_retrieve=True,
    )
    chunks = [
        RetrievalChunk(id="a", text="ok", score=0.9, metadata={"embedding_model_version": "v2"}),
        RetrievalChunk(id="b", text="stale", score=0.8, metadata={"embedding_model_version": "v1"}),
        RetrievalChunk(id="c", text="legacy", score=0.7, metadata={}),
    ]

    kept, filtered_count, warnings = filter_chunks_by_embedding_version(chunks, profile=profile)

    assert [chunk.id for chunk in kept] == ["a", "c"]
    assert filtered_count == 1
    assert warnings == ["filtered_mismatched_chunks:1"]
