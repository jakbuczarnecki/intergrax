from __future__ import annotations

from intergrax.runtime.architecture.retrieval_security import (
    RetrievalDocumentSignal,
    RetrievalTrustLevel,
    evaluate_retrieval_poisoning,
)


def test_retrieval_security_quarantines_low_trust_documents() -> None:
    report = evaluate_retrieval_poisoning(
        signals=[
            RetrievalDocumentSignal(
                document_id="doc-1",
                trust_score=0.2,
                source_ref="kb/unverified",
            )
        ]
    )
    assert report.decisions[0].quarantined is True
    assert report.decisions[0].trust_level == RetrievalTrustLevel.QUARANTINED


def test_retrieval_security_marks_mid_trust_as_review() -> None:
    report = evaluate_retrieval_poisoning(
        signals=[
            RetrievalDocumentSignal(
                document_id="doc-2",
                trust_score=0.6,
                source_ref="kb/external",
            )
        ]
    )
    assert report.decisions[0].trust_level == RetrievalTrustLevel.REVIEW
