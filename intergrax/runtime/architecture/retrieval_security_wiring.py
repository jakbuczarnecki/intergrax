# © Artur Czarnecki. All rights reserved.

"""Retrieval poisoning defense wiring for the RAG execution path (Phase V-REM-SEC.2)."""

from __future__ import annotations

from intergrax.runtime.architecture.retrieval_security import (
    RetrievalDocumentSignal,
    RetrievalTrustLevel,
    evaluate_retrieval_poisoning,
)
from intergrax.runtime.architecture.retrieval_security import RetrievalPoisoningInputChunk


def filter_retrieved_chunks_for_poisoning(
    chunks: list[RetrievalPoisoningInputChunk],
    *,
    quarantine_threshold: float = 0.40,
    review_threshold: float = 0.70,
) -> tuple[list[RetrievalPoisoningInputChunk], list[str]]:
    """Drop quarantined chunks and return manual-review warnings."""
    if not chunks:
        return [], []

    signals: list[RetrievalDocumentSignal] = []
    for chunk in chunks:
        signals.append(
            RetrievalDocumentSignal(
                document_id=chunk.id,
                trust_score=chunk.score,
                source_ref=chunk.source_ref,
            )
        )

    report = evaluate_retrieval_poisoning(
        signals=signals,
        quarantine_threshold=quarantine_threshold,
        review_threshold=review_threshold,
    )
    quarantined_ids = {
        decision.document_id
        for decision in report.decisions
        if decision.trust_level == RetrievalTrustLevel.QUARANTINED
    }
    filtered = [chunk for chunk in chunks if chunk.id not in quarantined_ids]
    warnings = [
        f"{decision.document_id}: {', '.join(decision.reasons)}"
        for decision in report.decisions
        if decision.trust_level == RetrievalTrustLevel.REVIEW
    ]
    return filtered, warnings
