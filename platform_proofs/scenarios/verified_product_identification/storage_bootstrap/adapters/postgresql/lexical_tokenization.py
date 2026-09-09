"""Product-aware lexical tokenization for VPI BM25 retrieval."""

from __future__ import annotations

import re

_LEXICAL_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*",
    re.UNICODE,
)


def tokenize_lexical_document(text: str) -> tuple[str, ...]:
    """Tokenize lexical text preserving model codes, hyphens, and mixed alphanumerics."""
    normalized = text.casefold()
    return tuple(_LEXICAL_TOKEN_PATTERN.findall(normalized))
