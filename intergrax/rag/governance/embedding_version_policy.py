# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Embedding model version mismatch policy for ingest and retrieve (M-RAG.31)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, TypeVar

from intergrax.rag.profiles.rag_profile import RagProfile

TChunk = TypeVar("TChunk")

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_VERSION_METADATA_KEY = "embedding_model_version"

ReindexQueueHook = Callable[["ReindexQueueRequest"], None]

_reindex_queue_hooks: list[ReindexQueueHook] = []


@dataclass(frozen=True)
class ReindexQueueRequest:
    source_path: str
    current_version: Optional[str]
    target_version: str
    reason: str


@dataclass
class EmbeddingVersionPolicyResult:
    warnings: List[str] = field(default_factory=list)
    reindex_enqueued: bool = False


def register_reindex_queue_hook(callback: ReindexQueueHook) -> None:
    """Register a Tier-3/workflow hook for stale-index reindex requests."""
    _reindex_queue_hooks.append(callback)


def clear_reindex_queue_hooks() -> None:
    _reindex_queue_hooks.clear()


def normalize_embedding_model_version(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def evaluate_ingest_embedding_version(
    *,
    profile: RagProfile,
    base_metadata: Mapping[str, Any],
    source_path: str,
    indexed_version_hint: Optional[str] = None,
) -> EmbeddingVersionPolicyResult:
    """
    Warn when ingest metadata or indexed corpus version disagrees with profile.

    When a mismatch is detected, optional reindex hooks are notified.
    """
    result = EmbeddingVersionPolicyResult()
    if not profile.embedding_version_warn_on_ingest:
        return result

    target = normalize_embedding_model_version(profile.embedding_model_version)
    if target is None:
        return result

    incoming = normalize_embedding_model_version(
        base_metadata.get(EMBEDDING_MODEL_VERSION_METADATA_KEY)
    )
    if incoming is not None and incoming != target:
        result.warnings.append(f"incoming_metadata_version_mismatch:{incoming}!={target}")
        result.reindex_enqueued = _notify_reindex_needed(
            source_path=source_path,
            current_version=incoming,
            target_version=target,
            reason="incoming_metadata_version_mismatch",
        ) or result.reindex_enqueued

    indexed = normalize_embedding_model_version(indexed_version_hint)
    if indexed is not None and indexed != target:
        result.warnings.append(f"indexed_version_mismatch:{indexed}!={target}")
        result.reindex_enqueued = _notify_reindex_needed(
            source_path=source_path,
            current_version=indexed,
            target_version=target,
            reason="indexed_version_mismatch",
        ) or result.reindex_enqueued

    for warning in result.warnings:
        logger.warning(
            "embedding_version_policy source=%s warning=%s",
            source_path,
            warning,
        )
    return result


def filter_chunks_by_embedding_version(
    chunks: Sequence[TChunk],
    *,
    profile: RagProfile,
) -> tuple[list[TChunk], int, list[str]]:
    """
    Drop chunks whose ``embedding_model_version`` metadata disagrees with profile.

    Chunks without the metadata key are retained (legacy index compatibility).
    """
    if not profile.embedding_version_filter_on_retrieve:
        return list(chunks), 0, []

    target = normalize_embedding_model_version(profile.embedding_model_version)
    if target is None:
        return list(chunks), 0, []

    kept: list[TChunk] = []
    filtered_count = 0
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        chunk_version = normalize_embedding_model_version(
            metadata.get(EMBEDDING_MODEL_VERSION_METADATA_KEY)
        )
        if chunk_version is not None and chunk_version != target:
            filtered_count += 1
            continue
        kept.append(chunk)

    warnings: list[str] = []
    if filtered_count:
        warnings.append(f"filtered_mismatched_chunks:{filtered_count}")
    return kept, filtered_count, warnings


def _notify_reindex_needed(
    *,
    source_path: str,
    current_version: Optional[str],
    target_version: str,
    reason: str,
) -> bool:
    if not _reindex_queue_hooks:
        return False
    request = ReindexQueueRequest(
        source_path=source_path,
        current_version=current_version,
        target_version=target_version,
        reason=reason,
    )
    for hook in _reindex_queue_hooks:
        hook(request)
    return True
