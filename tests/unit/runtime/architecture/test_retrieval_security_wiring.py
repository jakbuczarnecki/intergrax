from __future__ import annotations

from intergrax.runtime.architecture.retrieval_security_wiring import (
    filter_retrieved_chunks_for_poisoning,
)
from intergrax.runtime.nexus.context.context_builder import RetrievedChunk


def test_filter_retrieved_chunks_quarantines_low_trust_scores() -> None:
    chunks = [
        RetrievedChunk(id="trusted", text="ok", metadata={}, score=0.95),
        RetrievedChunk(id="poisoned", text="bad", metadata={}, score=0.10),
    ]
    filtered, warnings = filter_retrieved_chunks_for_poisoning(chunks)
    assert [chunk.id for chunk in filtered] == ["trusted"]
    assert warnings == []
