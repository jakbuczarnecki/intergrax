# © Artur Czarnecki. All rights reserved.

"""Retrieval poisoning defense wiring for the RAG execution path (Phase V-REM-SEC.2)."""

from __future__ import annotations

from intergrax.runtime.architecture.retrieval_security import (
    RetrievalDocumentSignal,
    RetrievalTrustLevel,
    evaluate_retrieval_poisoning,
)
from intergrax.runtime.nexus.context.context_builder import RetrievedChunk


def filter_retrieved_chunks_for_poisoning(
    chunks: list[RetrievedChunk],
    *,
    quarantine_threshold: float = 0.40,
    review_threshold: float = 0.70,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Drop quarantined chunks and return manual-review warnings."""
    if not chunks:
        return [], []

    signals: list[RetrievalDocumentSignal] = []
    for chunk in chunks:
        source_ref = chunk.metadata.get("source_ref")
        signals.append(
            RetrievalDocumentSignal(
                document_id=chunk.id,
                trust_score=chunk.score,
                source_ref=str(source_ref if source_ref is not None else chunk.id),
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
