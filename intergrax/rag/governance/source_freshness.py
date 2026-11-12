# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RAG source-freshness contract (PDC-D2-R005, OD-PDC-SCOPE-06 = A).

Indexed documents carry the source revision and/or modification timestamp they were indexed from.
Retrieval-side callers compare that stamp with the observed state of the source and receive a
typed :class:`SourceFreshnessVerdict` per source. Stale sources are handed to the existing
embedding-version reindex hook (``register_reindex_queue_hook``); this module adds no retrieval
engine, queue or scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from intergrax.rag.governance.embedding_version_policy import _notify_reindex_needed
from intergrax.utils import attribute_access

__all__ = [
    "RetrievalFreshnessAssessment",
    "SOURCE_MODIFIED_AT_METADATA_KEY",
    "SOURCE_REVISION_METADATA_KEY",
    "SourceFreshnessObservation",
    "SourceFreshnessStatus",
    "SourceFreshnessVerdict",
    "assess_retrieval_freshness",
    "evaluate_source_freshness",
    "observe_file_source",
    "stamp_source_freshness",
]

SOURCE_REVISION_METADATA_KEY = "source_revision"
SOURCE_MODIFIED_AT_METADATA_KEY = "source_modified_at"
_SOURCE_ID_METADATA_KEYS = ("source_path", "source", "source_id")

_REINDEX_REASON_BY_VERDICT_REASON = {
    "revision_mismatch": "source_revision_stale",
    "newer_modification_timestamp": "source_timestamp_stale",
    "source_missing": "source_missing",
}


class SourceFreshnessStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceFreshnessObservation:
    """Current state of a source as observed by the caller (file, URL, repository, ...)."""

    source_id: str
    revision: str | None = None
    modified_at: datetime | None = None
    exists: bool = True


@dataclass(frozen=True, slots=True)
class SourceFreshnessVerdict:
    """Typed staleness verdict for one indexed source."""

    source_id: str
    status: SourceFreshnessStatus
    reason: str
    indexed_revision: str | None = None
    current_revision: str | None = None
    indexed_modified_at: datetime | None = None
    current_modified_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RetrievalFreshnessAssessment:
    """One verdict per distinct source of a retrieval result."""

    verdicts: tuple[SourceFreshnessVerdict, ...]
    stale_source_ids: tuple[str, ...]
    reindex_requested: bool


def _normalize_revision(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _aware(value)
    text = _normalize_revision(value)
    if text is None:
        return None
    try:
        return _aware(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def stamp_source_freshness(
    base_metadata: Mapping[str, Any],
    *,
    revision: str | None = None,
    modified_at: datetime | None = None,
) -> dict[str, Any]:
    """Return ``base_metadata`` plus the source revision / timestamp the document is indexed from."""
    stamped = dict(base_metadata)
    normalized = _normalize_revision(revision)
    if normalized is not None:
        stamped[SOURCE_REVISION_METADATA_KEY] = normalized
    moment = _aware(modified_at)
    if moment is not None:
        stamped[SOURCE_MODIFIED_AT_METADATA_KEY] = moment.isoformat()
    return stamped


def observe_file_source(path: Path | str) -> SourceFreshnessObservation:
    """Observe a file source: revision is ``<mtime_ns>-<size>``, timestamp is the mtime (UTC)."""
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return SourceFreshnessObservation(source_id=str(path), exists=False)
    return SourceFreshnessObservation(
        source_id=str(path),
        revision=f"{stat.st_mtime_ns}-{stat.st_size}",
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )


def _chunk_metadata(chunk: Any) -> Mapping[str, Any]:
    metadata = attribute_access.optional(chunk, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _chunk_source_id(chunk: Any) -> str:
    metadata = _chunk_metadata(chunk)
    for key in _SOURCE_ID_METADATA_KEYS:
        value = _normalize_revision(metadata.get(key))
        if value is not None:
            return value
    return ""


def evaluate_source_freshness(
    chunk: Any,
    *,
    current: SourceFreshnessObservation,
) -> SourceFreshnessVerdict:
    """Compare an indexed chunk's freshness stamp with the observed source state."""
    metadata = _chunk_metadata(chunk)
    indexed_revision = _normalize_revision(metadata.get(SOURCE_REVISION_METADATA_KEY))
    indexed_at = _parse_timestamp(metadata.get(SOURCE_MODIFIED_AT_METADATA_KEY))
    current_revision = _normalize_revision(current.revision)
    current_at = _aware(current.modified_at)

    def verdict(status: SourceFreshnessStatus, reason: str) -> SourceFreshnessVerdict:
        return SourceFreshnessVerdict(
            source_id=current.source_id,
            status=status,
            reason=reason,
            indexed_revision=indexed_revision,
            current_revision=current_revision,
            indexed_modified_at=indexed_at,
            current_modified_at=current_at,
        )

    if not current.exists:
        return verdict(SourceFreshnessStatus.STALE, "source_missing")
    if indexed_revision is not None and current_revision is not None:
        if indexed_revision != current_revision:
            return verdict(SourceFreshnessStatus.STALE, "revision_mismatch")
        return verdict(SourceFreshnessStatus.FRESH, "revision_match")
    if indexed_at is not None and current_at is not None:
        if current_at > indexed_at:
            return verdict(SourceFreshnessStatus.STALE, "newer_modification_timestamp")
        return verdict(SourceFreshnessStatus.FRESH, "timestamp_not_newer")
    return verdict(SourceFreshnessStatus.UNKNOWN, "no_comparable_freshness_metadata")


_SEVERITY = {
    SourceFreshnessStatus.FRESH: 0,
    SourceFreshnessStatus.UNKNOWN: 1,
    SourceFreshnessStatus.STALE: 2,
}


def assess_retrieval_freshness(
    chunks: Sequence[Any],
    *,
    observations: Mapping[str, SourceFreshnessObservation],
    request_reindex: bool = True,
) -> RetrievalFreshnessAssessment:
    """Evaluate a retrieval result and queue reindex for stale sources via the existing hook.

    One verdict per distinct source; when chunks of a source disagree the worst verdict wins.
    A source without an observation is UNKNOWN (never silently fresh).
    """
    by_source: dict[str, SourceFreshnessVerdict] = {}
    for chunk in chunks:
        source_id = _chunk_source_id(chunk)
        observation = observations.get(source_id) if source_id else None
        if observation is None:
            verdict = SourceFreshnessVerdict(
                source_id=source_id,
                status=SourceFreshnessStatus.UNKNOWN,
                reason="source_not_observed" if source_id else "chunk_source_unidentified",
            )
        else:
            verdict = evaluate_source_freshness(chunk, current=observation)
        existing = by_source.get(source_id)
        if existing is None or _SEVERITY[verdict.status] > _SEVERITY[existing.status]:
            by_source[source_id] = verdict

    verdicts = tuple(by_source.values())
    stale = tuple(v.source_id for v in verdicts if v.status is SourceFreshnessStatus.STALE)
    reindex_requested = False
    if request_reindex:
        for verdict in verdicts:
            if verdict.status is not SourceFreshnessStatus.STALE:
                continue
            target = (
                verdict.current_revision
                or (verdict.current_modified_at.isoformat() if verdict.current_modified_at else None)
                or "unknown"
            )
            reindex_requested = (
                _notify_reindex_needed(
                    source_path=verdict.source_id,
                    current_version=verdict.indexed_revision
                    or (verdict.indexed_modified_at.isoformat() if verdict.indexed_modified_at else None),
                    target_version=target,
                    reason=_REINDEX_REASON_BY_VERDICT_REASON.get(verdict.reason, "source_stale"),
                )
                or reindex_requested
            )
    return RetrievalFreshnessAssessment(
        verdicts=verdicts,
        stale_source_ids=stale,
        reindex_requested=reindex_requested,
    )
