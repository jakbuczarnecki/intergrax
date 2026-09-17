# © Artur Czarnecki. All rights reserved.

"""Deterministic fragment canonicalization before exact dedup."""

from __future__ import annotations

from intergrax.context.contracts import (
    canonicalize_fragment_content,
    content_hash_for_text,
    ContextFragment,
    replace_context_fragment,
)


def canonicalize_fragment_for_policy(fragment: ContextFragment) -> ContextFragment:
    canonical_content = canonicalize_fragment_content(fragment.content)
    if canonical_content == fragment.content and fragment.content_hash:
        return fragment
    return replace_context_fragment(
        fragment,
        content=canonical_content,
        content_hash=content_hash_for_text(canonical_content),
    )
