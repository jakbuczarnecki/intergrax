"""Character-length profiling for qualification semantic texts."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    TextLengthStatistics,
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def profile_text_lengths(
    texts: Sequence[str],
    *,
    token_counts: Sequence[int] | None = None,
) -> TextLengthStatistics:
    if not texts:
        msg = "texts must not be empty"
        raise ValueError(msg)
    char_lengths = [len(text) for text in texts]
    token_mean: float | None = None
    token_p50: float | None = None
    token_p95: float | None = None
    token_max: int | None = None
    if token_counts is not None and len(token_counts) == len(texts):
        token_mean = statistics.fmean(token_counts)
        token_p50 = _percentile([float(value) for value in token_counts], 0.50)
        token_p95 = _percentile([float(value) for value in token_counts], 0.95)
        token_max = max(token_counts)
    return TextLengthStatistics(
        character_min=min(char_lengths),
        character_mean=statistics.fmean(char_lengths),
        character_p50=_percentile([float(value) for value in char_lengths], 0.50),
        character_p95=_percentile([float(value) for value in char_lengths], 0.95),
        character_max=max(char_lengths),
        token_mean=token_mean,
        token_p50=token_p50,
        token_p95=token_p95,
        token_max=token_max,
    )
