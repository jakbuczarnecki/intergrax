# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed C1 retrieval evidence contracts for instrumentation boundaries (DIAG-FUNCTIONAL-Q1-R1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalCandidateFact:
    """Provider-ranked retrieval candidate observed by instrumentation."""

    artifact_ref: str
    rank: int
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalSelectionFact:
    """Pipeline selection observed at the retrieval decision point."""

    selected_artifact_ref: str
    candidate_count: int
    selection_reason: str = "top_ranked"


@dataclass(frozen=True, slots=True)
class RetrievalEvidenceItem:
    """Minimal typed adapter input for C1 retrieval instrumentation."""

    chunk_id: str | None
    source_path: str | None
    score: float | None


def artifact_ref_from_retrieval_item(item: RetrievalEvidenceItem) -> str:
    """Deterministic identity mapping — not a relevance or selection heuristic."""
    if item.chunk_id is not None and item.chunk_id.strip():
        return f"chunk:{item.chunk_id.strip()}"
    if item.source_path is not None and item.source_path.strip():
        leaf = item.source_path.replace("\\", "/").rsplit("/", 1)[-1]
        if leaf:
            return f"source:{leaf}"
    return "chunk:unknown"


def parse_retrieval_evidence_items(
    evidence: list[dict[str, object]],
) -> tuple[RetrievalEvidenceItem, ...]:
    items: list[RetrievalEvidenceItem] = []
    for raw in evidence:
        chunk_raw = raw.get("chunk_id")
        source_raw = raw.get("source_path") or raw.get("source")
        score_raw = raw.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None
        items.append(
            RetrievalEvidenceItem(
                chunk_id=(
                    str(chunk_raw).strip()
                    if chunk_raw is not None and str(chunk_raw).strip()
                    else None
                ),
                source_path=(
                    str(source_raw).strip()
                    if isinstance(source_raw, str) and source_raw.strip()
                    else None
                ),
                score=score,
            ),
        )
    return tuple(items)


def top_ranked_artifact_ref(items: tuple[RetrievalEvidenceItem, ...]) -> str:
    """Pipeline top-1 selection policy for ranked retrieval results."""
    if not items:
        return "chunk:unknown"
    return artifact_ref_from_retrieval_item(items[0])


__all__ = [
    "RetrievalCandidateFact",
    "RetrievalEvidenceItem",
    "RetrievalSelectionFact",
    "artifact_ref_from_retrieval_item",
    "parse_retrieval_evidence_items",
    "top_ranked_artifact_ref",
]
