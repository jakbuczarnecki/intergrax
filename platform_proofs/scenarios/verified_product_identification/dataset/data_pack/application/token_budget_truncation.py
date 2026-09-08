"""Tokenizer-bounded document text truncation for embedding input policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def truncate_to_token_limit(
    text: str,
    *,
    token_limit: int,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> str:
    if token_limit <= 0:
        msg = "token_limit must be > 0"
        raise ValueError(msg)
    if not text:
        return text
    token_ids = encode(text)
    if len(token_ids) <= token_limit:
        return text
    return decode(token_ids[:token_limit])
