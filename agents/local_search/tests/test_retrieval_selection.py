# © Artur Czarnecki. All rights reserved.

"""Unit tests for search-owned retrieval selection contracts."""

from __future__ import annotations

import pytest

from local_search.retrieval_selection import (
    SearchRetrievalCandidate,
    artifact_ref_from_candidate,
    candidates_from_formatted_evidence,
    select_top_ranked_candidate,
)

pytestmark = pytest.mark.unit


def test_source_leaf_fallback_when_chunk_id_missing() -> None:
    candidate = SearchRetrievalCandidate(
        chunk_id=None,
        source_path="/data/incident-report.md",
        score=None,
    )
    assert artifact_ref_from_candidate(candidate) == "source:incident-report.md"


def test_unknown_when_identity_missing() -> None:
    candidate = SearchRetrievalCandidate(chunk_id=None, source_path=None, score=None)
    assert artifact_ref_from_candidate(candidate) == "chunk:unknown"


def test_candidates_from_formatted_evidence_preserves_rank_order() -> None:
    candidates = candidates_from_formatted_evidence(
        [
            {"chunk_id": "a"},
            {"chunk_id": "b"},
        ],
    )
    selection = select_top_ranked_candidate(candidates)
    assert selection.selected_artifact_ref == "chunk:a"
    assert selection.candidate_count == 2


def test_select_top_ranked_candidate_empty() -> None:
    selection = select_top_ranked_candidate(())
    assert selection.selected_artifact_ref == "chunk:unknown"
    assert selection.candidate_count == 0
