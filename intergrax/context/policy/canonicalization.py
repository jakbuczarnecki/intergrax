# © Artur Czarnecki. All rights reserved.

"""Deterministic fragment canonicalization before exact dedup."""

from __future__ import annotations

from intergrax.context.contracts import (
    canonicalize_fragment_content,
    content_hash_for_text,
    ContextFragment,
    ContextFragmentSource,
    replace_context_fragment,
)
from intergrax.context.session_history import session_history_content_hash_for_fragment


def canonicalize_fragment_for_policy(fragment: ContextFragment) -> ContextFragment:
    canonical_content = canonicalize_fragment_content(fragment.content)
    if fragment.source is ContextFragmentSource.SESSION_HISTORY:
        candidate = replace_context_fragment(fragment, content=canonical_content)
        expected_hash = session_history_content_hash_for_fragment(candidate)
        metadata_hash = fragment.metadata.get("content_hash")
        if (
            canonical_content == fragment.content
            and fragment.content_hash == expected_hash
            and metadata_hash == expected_hash
        ):
            return fragment
        metadata = dict(fragment.metadata)
        metadata["content_hash"] = expected_hash
        return replace_context_fragment(
            candidate,
            content_hash=expected_hash,
            metadata=metadata,
        )
    if canonical_content == fragment.content and fragment.content_hash:
        return fragment
    return replace_context_fragment(
        fragment,
        content=canonical_content,
        content_hash=content_hash_for_text(canonical_content),
    )
