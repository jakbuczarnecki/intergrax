# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sync vs async ingest policy helpers (M-RAG.26)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.profiles.rag_profile import RagProfile

SYNC_INGEST_SIZE_EXCEEDED_REASON = "sync_ingest_size_exceeded"
SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON = "semantic_chunking_size_exceeded"


def source_file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size)


def sync_ingest_allowed(*, path: Path, profile: RagProfile) -> tuple[bool, str, int]:
    """
    Return whether the sync ingest path may load ``path`` into memory.

    ``(allowed, reason, file_size_bytes)``
    """
    size = source_file_size_bytes(path)
    max_bytes = int(profile.sync_ingest_max_bytes)
    if max_bytes > 0 and size > max_bytes:
        return (
            False,
            f"{SYNC_INGEST_SIZE_EXCEEDED_REASON}:{size}>{max_bytes}",
            size,
        )
    return True, "ok", size


def semantic_chunking_allowed(
    *,
    docs: Sequence[KnowledgeDocument],
    strategy_id: str,
    profile: RagProfile,
) -> tuple[bool, str, int]:
    """
    Reject semantic chunking when loaded document text exceeds the profile char budget.

    Avoids O(n) sentence embedding cost in ``SemanticChunkingStrategy`` for oversized
    documents. Returns ``(allowed, reason, offending_char_count)``.
    """
    if strategy_id != "semantic":
        return True, "ok", 0
    max_chars = int(profile.semantic_chunking_max_chars)
    if max_chars <= 0:
        return True, "ok", 0
    for doc in docs:
        chars = len(doc.content or "")
        if chars > max_chars:
            return (
                False,
                f"{SEMANTIC_CHUNKING_SIZE_EXCEEDED_REASON}:{chars}>{max_chars}",
                chars,
            )
    largest = max((len(doc.content or "") for doc in docs), default=0)
    return True, "ok", largest


def build_ingest_idempotency_key(
    *,
    source_path: str,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
    explicit_key: str | None = None,
) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    normalized = str(Path(source_path).expanduser().resolve())
    payload = "|".join([normalized, tenant_id or "", workspace_id or ""])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"rag-ingest-{digest}"
