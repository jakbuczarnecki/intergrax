# © Artur Czarnecki. All rights reserved.

"""Unit tests for C1 retrieval evidence contracts."""

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.c1_retrieval_evidence import (
    artifact_ref_from_retrieval_item,
    parse_retrieval_evidence_items,
    top_ranked_artifact_ref,
    RetrievalEvidenceItem,
)

pytestmark = pytest.mark.unit


def test_source_leaf_fallback_when_chunk_id_missing() -> None:
    item = RetrievalEvidenceItem(chunk_id=None, source_path="/data/incident-report.md", score=None)
    assert artifact_ref_from_retrieval_item(item) == "source:incident-report.md"


def test_unknown_when_identity_missing() -> None:
    item = RetrievalEvidenceItem(chunk_id=None, source_path=None, score=None)
    assert artifact_ref_from_retrieval_item(item) == "chunk:unknown"


def test_parse_retrieval_evidence_items_preserves_rank_order() -> None:
    items = parse_retrieval_evidence_items(
        [
            {"chunk_id": "a"},
            {"chunk_id": "b"},
        ],
    )
    assert top_ranked_artifact_ref(items) == "chunk:a"
