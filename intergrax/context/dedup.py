# © Artur Czarnecki. All rights reserved.

"""Fragment deduplication by content hash (CE-10.2)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragment
from intergrax.context.policy.exact_dedup import _exact_identity_key as _dedup_identity_key
from intergrax.context.policy.exact_dedup import exact_dedup_fragments


def dedup_fragments_by_hash(
    fragments: list[ContextFragment],
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
    """Return unique fragments and dropped duplicates with suppression reason."""
    kept, dropped, _audit = exact_dedup_fragments(fragments)
    return kept, dropped
