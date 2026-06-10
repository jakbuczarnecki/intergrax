# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Formal citation model for RAG retrieval engine output (M-RAG.29)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from intergrax.rag.retrieval.retrieval_result import RetrievalChunk

_SOURCE_LABEL_KEYS = ("title", "source", "url", "doc_id", "file", "document_id")
_SOURCE_ID_KEYS = ("doc_id", "document_id", "source_id", "source", "file")
_URL_KEYS = ("url", "source_url", "link")
_PAGE_KEYS = ("page", "page_number")


@dataclass(frozen=True)
class Citation:
    """Structured provenance for a retrieved chunk."""

    chunk_id: str
    source_id: str
    source_type: str = "vectorstore"
    source_label: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    excerpt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def citation_from_chunk(chunk: RetrievalChunk, *, excerpt_max_chars: int = 240) -> Citation:
    metadata = dict(chunk.metadata or {})
    source_label = _first_metadata_str(metadata, _SOURCE_LABEL_KEYS)
    source_id = _first_metadata_str(metadata, _SOURCE_ID_KEYS) or chunk.id
    url = _first_metadata_str(metadata, _URL_KEYS)
    page = _parse_page(metadata)
    excerpt = _build_excerpt(chunk.text, max_chars=excerpt_max_chars)
    return Citation(
        chunk_id=chunk.id,
        source_id=source_id,
        source_label=source_label,
        url=url,
        page=page,
        score=float(chunk.score),
        excerpt=excerpt,
        metadata=metadata,
    )


def citations_from_chunks(
    chunks: Sequence[RetrievalChunk],
    *,
    excerpt_max_chars: int = 240,
) -> List[Citation]:
    return [citation_from_chunk(chunk, excerpt_max_chars=excerpt_max_chars) for chunk in chunks]


def _first_metadata_str(metadata: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _parse_page(metadata: Dict[str, Any]) -> Optional[int]:
    for key in _PAGE_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_excerpt(text: str, *, max_chars: int) -> Optional[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip() + "…"
