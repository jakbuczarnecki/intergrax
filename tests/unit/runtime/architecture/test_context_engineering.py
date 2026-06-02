from __future__ import annotations

import pytest

from intergrax.runtime.architecture.context_engineering import (
    ContextChunkSignal,
    ContextQualityThresholds,
    deduplicate_context_chunks,
    evaluate_context_engineering,
)


def test_deduplicate_context_chunks_suppresses_duplicate_hash() -> None:
    unique, suppressed = deduplicate_context_chunks(
        [
            ContextChunkSignal(
                chunk_id="a",
                content_hash="same",
                relevance_score=0.9,
                freshness_score=0.9,
                confidence_score=0.9,
            ),
            ContextChunkSignal(
                chunk_id="b",
                content_hash="same",
                relevance_score=0.8,
                freshness_score=0.8,
                confidence_score=0.8,
            ),
        ]
    )
    assert len(unique) == 1
    assert suppressed == ["b"]


def test_context_engineering_fails_low_quality_chunk() -> None:
    report = evaluate_context_engineering(
        chunks=[
            ContextChunkSignal(
                chunk_id="low",
                content_hash="hash-low",
                relevance_score=0.4,
                freshness_score=0.4,
                confidence_score=0.4,
            )
        ],
        thresholds=ContextQualityThresholds(),
    )
    assert report.records[0].passed is False
    assert report.records[0].reasons


def test_context_engineering_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="range"):
        ContextChunkSignal(
            chunk_id="bad",
            content_hash="hash",
            relevance_score=1.5,
            freshness_score=0.5,
            confidence_score=0.5,
        )
