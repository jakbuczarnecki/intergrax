from __future__ import annotations

from intergrax.runtime.architecture.retrieval_effectiveness import (
    RetrievalJudgment,
    evaluate_retrieval_effectiveness,
)


def test_retrieval_effectiveness_computes_precision_and_recall() -> None:
    report = evaluate_retrieval_effectiveness(
        [
            RetrievalJudgment(
                query_id="q1",
                relevant_document_ids=["doc-1", "doc-2"],
                retrieved_document_ids=["doc-1", "doc-3", "doc-2"],
            )
        ],
        k=3,
    )
    assert report.metrics[0].precision_at_k == 2 / 3
    assert report.metrics[0].recall_at_k == 1.0
