# © Artur Czarnecki. All rights reserved.

"""Fragment deduplication by content hash (CE-10.2)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragment, ContextFragmentSource


def _dedup_identity_key(fragment: ContextFragment) -> str:
    if fragment.source is ContextFragmentSource.SESSION_HISTORY:
        return (
            f"session_history:{fragment.source_id}:{fragment.content_hash or ''}"
        )
    return fragment.content_hash or fragment.fragment_id


def dedup_fragments_by_hash(
    fragments: list[ContextFragment],
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
    """Return unique fragments and dropped duplicates with suppression reason."""
    seen: set[str] = set()
    kept: list[ContextFragment] = []
    dropped: list[tuple[ContextFragment, str]] = []
    for fragment in fragments:
        key = _dedup_identity_key(fragment)
        if key in seen:
            dropped.append((fragment, "duplicate_content_hash"))
            continue
        seen.add(key)
        kept.append(fragment)
    return kept, dropped
