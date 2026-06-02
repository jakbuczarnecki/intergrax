# © Artur Czarnecki. All rights reserved.

"""Retrieval effectiveness evaluation contracts (Phase V-CE.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalJudgment(BaseModel):
    query_id: str
    relevant_document_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)


class RetrievalEffectivenessMetrics(BaseModel):
    query_id: str
    precision_at_k: float
    recall_at_k: float
    k: int


class RetrievalEffectivenessReport(BaseModel):
    schema_version: str = "1.0.0"
    k: int
    metrics: list[RetrievalEffectivenessMetrics] = Field(default_factory=list)
    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0


def evaluate_retrieval_effectiveness(
    judgments: list[RetrievalJudgment],
    *,
    k: int = 5,
) -> RetrievalEffectivenessReport:
    metrics: list[RetrievalEffectivenessMetrics] = []
    for judgment in judgments:
        retrieved_top_k = judgment.retrieved_document_ids[:k]
        relevant_set = set(judgment.relevant_document_ids)
        if not retrieved_top_k:
            precision = 0.0
        else:
            hits = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)
            precision = float(hits) / float(len(retrieved_top_k))
        recall = (
            float(sum(1 for doc_id in relevant_set if doc_id in retrieved_top_k))
            / float(len(relevant_set))
            if relevant_set
            else 0.0
        )
        metrics.append(
            RetrievalEffectivenessMetrics(
                query_id=judgment.query_id,
                precision_at_k=precision,
                recall_at_k=recall,
                k=k,
            )
        )
    mean_precision = sum(item.precision_at_k for item in metrics) / float(len(metrics)) if metrics else 0.0
    mean_recall = sum(item.recall_at_k for item in metrics) / float(len(metrics)) if metrics else 0.0
    return RetrievalEffectivenessReport(
        k=k,
        metrics=metrics,
        mean_precision_at_k=mean_precision,
        mean_recall_at_k=mean_recall,
    )
