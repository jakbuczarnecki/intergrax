# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Search-owned retrieval candidate contracts and top-1 selection policy (DIAG-FUNCTIONAL-Q1-R2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchRetrievalCandidate:
    """Provider-ranked retrieval candidate in search layer."""

    chunk_id: str | None
    source_path: str | None
    score: float | None


@dataclass(frozen=True, slots=True)
class SearchRetrievalSelection:
    """Actual retrieval selection produced by search policy."""

    selected_artifact_ref: str
    candidate_count: int
    selection_reason: str = "top_ranked"


def artifact_ref_from_candidate(candidate: SearchRetrievalCandidate) -> str:
    """Deterministic identity mapping — not a relevance or selection heuristic."""
    if candidate.chunk_id is not None and candidate.chunk_id.strip():
        return f"chunk:{candidate.chunk_id.strip()}"
    if candidate.source_path is not None and candidate.source_path.strip():
        leaf = candidate.source_path.replace("\\", "/").rsplit("/", 1)[-1]
        if leaf:
            return f"source:{leaf}"
    return "chunk:unknown"


def candidates_from_formatted_evidence(
    evidence: list[dict[str, object]],
) -> tuple[SearchRetrievalCandidate, ...]:
    """Typed adapter from formatted search evidence to ranked candidates."""
    items: list[SearchRetrievalCandidate] = []
    for raw in evidence:
        chunk_raw = raw.get("chunk_id")
        source_raw = raw.get("source_path") or raw.get("source")
        score_raw = raw.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None
        items.append(
            SearchRetrievalCandidate(
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


def select_top_ranked_candidate(
    candidates: tuple[SearchRetrievalCandidate, ...],
) -> SearchRetrievalSelection:
    """Pipeline top-1 selection policy for ranked retrieval results."""
    if not candidates:
        return SearchRetrievalSelection(
            selected_artifact_ref="chunk:unknown",
            candidate_count=0,
        )
    return SearchRetrievalSelection(
        selected_artifact_ref=artifact_ref_from_candidate(candidates[0]),
        candidate_count=len(candidates),
    )


__all__ = [
    "SearchRetrievalCandidate",
    "SearchRetrievalSelection",
    "artifact_ref_from_candidate",
    "candidates_from_formatted_evidence",
    "select_top_ranked_candidate",
]
