# © Artur Czarnecki. All rights reserved.

"""PDC-D2 R005: RAG source-freshness contract (OD-PDC-SCOPE-06 = A), inside rag/governance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from intergrax.rag.governance.embedding_version_policy import (
    ReindexQueueRequest,
    clear_reindex_queue_hooks,
    register_reindex_queue_hook,
)
from intergrax.rag.governance.source_freshness import (
    SOURCE_MODIFIED_AT_METADATA_KEY,
    SOURCE_REVISION_METADATA_KEY,
    SourceFreshnessObservation,
    SourceFreshnessStatus,
    assess_retrieval_freshness,
    evaluate_source_freshness,
    observe_file_source,
    stamp_source_freshness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)


@dataclass
class _Chunk:
    """Retrieved chunk shape consumed by the governance helpers (metadata-bearing)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture(autouse=True)
def _isolated_reindex_hooks() -> Any:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


def _chunk(source: str, **freshness: Any) -> _Chunk:
    metadata = stamp_source_freshness({"source_path": source}, **freshness)
    return _Chunk(text=f"text of {source}", metadata=metadata)


def test_ingest_stamp_carries_revision_and_timestamp_without_mutating_input() -> None:
    base = {"source_path": "docs/a.md", "keep": "me"}
    stamped = stamp_source_freshness(base, revision="rev-7", modified_at=_T0)
    assert stamped[SOURCE_REVISION_METADATA_KEY] == "rev-7"
    assert stamped[SOURCE_MODIFIED_AT_METADATA_KEY] == _T0.isoformat()
    assert stamped["keep"] == "me"
    assert SOURCE_REVISION_METADATA_KEY not in base


def test_stale_source_detected_by_revision_mismatch() -> None:
    verdict = evaluate_source_freshness(
        _chunk("docs/a.md", revision="rev-1"),
        current=SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2"),
    )
    assert verdict.status is SourceFreshnessStatus.STALE
    assert verdict.indexed_revision == "rev-1"
    assert verdict.current_revision == "rev-2"
    assert verdict.reason == "revision_mismatch"


def test_matching_revision_is_fresh_and_missing_metadata_is_unknown_not_fresh() -> None:
    fresh = evaluate_source_freshness(
        _chunk("docs/a.md", revision="rev-1"),
        current=SourceFreshnessObservation(source_id="docs/a.md", revision="rev-1"),
    )
    assert fresh.status is SourceFreshnessStatus.FRESH

    legacy = evaluate_source_freshness(
        _Chunk(text="legacy", metadata={"source_path": "docs/old.md"}),
        current=SourceFreshnessObservation(source_id="docs/old.md", revision="rev-9"),
    )
    assert legacy.status is SourceFreshnessStatus.UNKNOWN
    assert legacy.reason == "no_comparable_freshness_metadata"


def test_stale_source_detected_by_newer_modification_timestamp() -> None:
    chunk = _chunk("docs/b.md", modified_at=_T0)
    newer = SourceFreshnessObservation(source_id="docs/b.md", modified_at=_T0 + timedelta(hours=2))
    same = SourceFreshnessObservation(source_id="docs/b.md", modified_at=_T0)
    assert evaluate_source_freshness(chunk, current=newer).status is SourceFreshnessStatus.STALE
    assert evaluate_source_freshness(chunk, current=same).status is SourceFreshnessStatus.FRESH


def test_deleted_source_is_stale() -> None:
    verdict = evaluate_source_freshness(
        _chunk("docs/gone.md", revision="rev-1"),
        current=SourceFreshnessObservation(source_id="docs/gone.md", exists=False),
    )
    assert verdict.status is SourceFreshnessStatus.STALE
    assert verdict.reason == "source_missing"


def test_retrieval_assessment_returns_typed_verdicts_per_source_and_requests_reindex() -> None:
    queued: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(queued.append)
    chunks = [
        _chunk("docs/a.md", revision="rev-1"),
        _chunk("docs/a.md", revision="rev-1"),  # same source: one verdict
        _chunk("docs/b.md", revision="rev-5"),
    ]
    observations = {
        "docs/a.md": SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2"),
        "docs/b.md": SourceFreshnessObservation(source_id="docs/b.md", revision="rev-5"),
    }

    assessment = assess_retrieval_freshness(chunks, observations=observations)

    by_source = {v.source_id: v for v in assessment.verdicts}
    assert set(by_source) == {"docs/a.md", "docs/b.md"}
    assert by_source["docs/a.md"].status is SourceFreshnessStatus.STALE
    assert by_source["docs/b.md"].status is SourceFreshnessStatus.FRESH
    assert assessment.stale_source_ids == ("docs/a.md",)
    assert assessment.reindex_requested is True
    # Reindex goes through the existing embedding-version reindex hook, once, for the stale source.
    assert [(r.source_path, r.current_version, r.target_version, r.reason) for r in queued] == [
        ("docs/a.md", "rev-1", "rev-2", "source_revision_stale")
    ]


def test_assessment_without_a_hook_still_reports_stale_but_enqueues_nothing() -> None:
    assessment = assess_retrieval_freshness(
        [_chunk("docs/a.md", revision="rev-1")],
        observations={"docs/a.md": SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2")},
    )
    assert assessment.stale_source_ids == ("docs/a.md",)
    assert assessment.reindex_requested is False


def test_unobserved_source_is_unknown_never_silently_fresh() -> None:
    assessment = assess_retrieval_freshness(
        [_chunk("docs/a.md", revision="rev-1")],
        observations={},
    )
    assert assessment.verdicts[0].status is SourceFreshnessStatus.UNKNOWN
    assert assessment.verdicts[0].reason == "source_not_observed"
    assert assessment.stale_source_ids == ()


def test_real_file_edit_after_indexing_is_detected_as_stale(tmp_path: Path) -> None:
    source = tmp_path / "policy.md"
    source.write_text("version one", encoding="utf-8")
    indexed = observe_file_source(source)
    chunk = _Chunk(
        text="version one",
        metadata=stamp_source_freshness(
            {"source_path": str(source)},
            revision=indexed.revision,
            modified_at=indexed.modified_at,
        ),
    )
    unchanged = evaluate_source_freshness(chunk, current=observe_file_source(source))
    assert unchanged.status is SourceFreshnessStatus.FRESH

    source.write_text("version two, edited after indexing", encoding="utf-8")
    later = os.stat(source).st_mtime_ns + 5_000_000_000
    os.utime(source, ns=(later, later))
    edited = evaluate_source_freshness(chunk, current=observe_file_source(source))
    assert edited.status is SourceFreshnessStatus.STALE

    source.unlink()
    gone = evaluate_source_freshness(chunk, current=observe_file_source(source))
    assert gone.status is SourceFreshnessStatus.STALE
    assert gone.reason == "source_missing"
