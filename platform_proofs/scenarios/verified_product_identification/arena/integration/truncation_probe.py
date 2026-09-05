"""Tokenizer-based truncation profiling for arena candidates."""

from __future__ import annotations

from collections.abc import Sequence

from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    TruncationProfile,
)
from platform_proofs.scenarios.verified_product_identification.qualification.text_length_profile import (
    profile_text_lengths,
)


def profile_truncation_for_texts(
    *,
    model_name: str,
    texts: Sequence[str],
    max_supported_tokens: int,
) -> TruncationProfile:
    if max_supported_tokens <= 0:
        msg = "max_supported_tokens must be > 0"
        raise ValueError(msg)
    if not texts:
        msg = "texts must not be empty"
        raise ValueError(msg)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    token_counts = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]
    truncated_count = sum(1 for count in token_counts if count > max_supported_tokens)
    stats = profile_text_lengths(texts, token_counts=token_counts)
    percentage = (truncated_count / len(texts)) * 100.0
    return TruncationProfile(
        tokenizer_model=model_name,
        max_supported_tokens=max_supported_tokens,
        truncated_count=truncated_count,
        truncated_percentage=percentage,
        token_p50=float(stats.token_p50 or 0.0),
        token_p95=float(stats.token_p95 or 0.0),
        token_max=int(stats.token_max or 0),
    )
