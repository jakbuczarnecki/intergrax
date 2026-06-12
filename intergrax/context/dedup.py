# © Artur Czarnecki. All rights reserved.

"""Fragment deduplication by content hash (CE-10.2)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragment


def dedup_fragments_by_hash(
    fragments: list[ContextFragment],
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
    """Return unique fragments and dropped duplicates with suppression reason."""
    seen: set[str] = set()
    kept: list[ContextFragment] = []
    dropped: list[tuple[ContextFragment, str]] = []
    for fragment in fragments:
        key = fragment.content_hash or fragment.fragment_id
        if key in seen:
            dropped.append((fragment, "duplicate_content_hash"))
            continue
        seen.add(key)
        kept.append(fragment)
    return kept, dropped
