"""Canonical BGE-M3 tokenizer truncation for product semantic text."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.contracts import (
    RepresentationVariant,
)


@dataclass(frozen=True, slots=True)
class ProductRepresentationVariantPort:
    """Transform canonical semantic text to a bounded variant representation."""

    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    count_tokens: Callable[[str], int]

    def apply(self, canonical_semantic_text: str, variant: RepresentationVariant) -> str:
        token_limit = variant.token_limit()
        if token_limit is None:
            return canonical_semantic_text
        if not canonical_semantic_text:
            return canonical_semantic_text
        token_ids = self.encode(canonical_semantic_text)
        if len(token_ids) <= token_limit:
            return canonical_semantic_text
        return self.decode(token_ids[:token_limit])

    def count_tokens_for(self, text: str) -> int:
        if not text:
            return 0
        return self.count_tokens(text)


def truncate_to_token_limit(
    text: str,
    *,
    token_limit: int,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
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
