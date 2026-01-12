# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for websearch dedupe utilities.

This suite defines the behavioral contract for text normalization and
deduplication key generation used in the websearch pipeline.

Why this matters:
- Normalization must be deterministic and stable to prevent "flaky" dedupe results.
- The dedupe key must be stable across superficial text variations (case/whitespace),
  ensuring reliable duplicate detection and predictable pipeline behavior.
"""

from __future__ import annotations

import pytest

from intergrax.websearch.utils.dedupe import normalize_for_dedupe, simple_dedupe_key


pytestmark = pytest.mark.unit


def test_normalize_for_dedupe_none_returns_empty_string() -> None:
    """
    None input must be treated as empty string.

    This guarantees downstream code can safely normalize optional fields
    without additional None-checks.
    """
    assert normalize_for_dedupe(None) == ""


def test_normalize_for_dedupe_empty_string_returns_empty_string() -> None:
    """
    Empty input should remain empty after normalization.
    """
    assert normalize_for_dedupe("") == ""


def test_normalize_for_dedupe_strips_and_lowercases() -> None:
    """
    Normalization must:
    - strip leading/trailing whitespace,
    - lower-case the text.

    This ensures superficial formatting differences do not affect dedupe.
    """
    assert normalize_for_dedupe("  HeLLo  ") == "hello"


def test_normalize_for_dedupe_collapses_internal_whitespace() -> None:
    """
    Internal whitespace sequences must collapse to a single space.

    This prevents duplicates caused by formatting differences such as multiple spaces,
    newlines, or tabs.
    """
    assert normalize_for_dedupe("a   b") == "a b"
    assert normalize_for_dedupe("a\t\tb") == "a b"
    assert normalize_for_dedupe("a\n\nb") == "a b"
    assert normalize_for_dedupe("  a \n  b\t c  ") == "a b c"


def test_normalize_for_dedupe_is_idempotent() -> None:
    """
    Normalization must be idempotent:

    normalize(x) == normalize(normalize(x))

    Idempotency is a key invariant for stable pipelines and caching.
    """
    text = "  A \n  B\t C  "
    once = normalize_for_dedupe(text)
    twice = normalize_for_dedupe(once)
    assert twice == once


def test_simple_dedupe_key_is_stable_for_identical_normalized_text() -> None:
    """
    The dedupe key must be stable: the same input always yields the same key.
    """
    text = "Hello world"
    assert simple_dedupe_key(text) == simple_dedupe_key(text)


def test_simple_dedupe_key_matches_for_case_and_whitespace_variants() -> None:
    """
    Case differences and whitespace-only differences must not affect the dedupe key.

    This is the core contract: near-identical documents should collapse to the same key.
    """
    a = "Hello   world"
    b = "  hello world  "
    c = "HELLO\tWORLD"
    d = "hello\nworld"

    assert simple_dedupe_key(a) == simple_dedupe_key(b) == simple_dedupe_key(c) == simple_dedupe_key(d)


def test_simple_dedupe_key_differs_for_different_semantics() -> None:
    """
    Semantically different text must produce a different key (extremely high probability).

    Note:
    This is not a cryptographic proof test; it's a practical regression guardrail.
    """
    assert simple_dedupe_key("hello world") != simple_dedupe_key("hello worlds")


def test_simple_dedupe_key_for_none_equals_key_for_empty_string() -> None:
    """
    None is normalized to empty string; therefore the dedupe key must match.
    """
    assert simple_dedupe_key(None) == simple_dedupe_key("")
