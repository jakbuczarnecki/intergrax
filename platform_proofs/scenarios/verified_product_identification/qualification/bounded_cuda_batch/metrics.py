"""Pure metric helpers for bounded CUDA batch throughput qualification."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    FULL_DATASET_RECORD_COUNT,
    EmbeddingTimeProjection,
    TokenProfile,
)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compute_token_profile(token_counts: Sequence[int]) -> TokenProfile:
    if not token_counts:
        msg = "token_counts must not be empty"
        raise ValueError(msg)
    floating = [float(value) for value in token_counts]
    return TokenProfile(
        total_tokens=sum(token_counts),
        average_tokens_per_record=statistics.fmean(token_counts),
        p50_tokens=percentile(floating, 0.50),
        p95_tokens=percentile(floating, 0.95),
        max_tokens=max(token_counts),
    )


def compute_throughput_metrics(
    *,
    record_count: int,
    total_tokens: int,
    wall_clock_seconds: float,
) -> tuple[float, float, float]:
    if record_count <= 0:
        msg = "record_count must be > 0"
        raise ValueError(msg)
    if wall_clock_seconds <= 0.0:
        msg = "wall_clock_seconds must be > 0"
        raise ValueError(msg)
    records_per_second = record_count / wall_clock_seconds
    tokens_per_second = total_tokens / wall_clock_seconds
    average_milliseconds_per_record = (wall_clock_seconds / record_count) * 1000.0
    return records_per_second, tokens_per_second, average_milliseconds_per_record


def compute_vram_headroom_fraction(
    *,
    gpu_total_memory_bytes: int,
    peak_cuda_allocated_bytes: int,
) -> float:
    if gpu_total_memory_bytes <= 0:
        msg = "gpu_total_memory_bytes must be > 0"
        raise ValueError(msg)
    remaining = gpu_total_memory_bytes - peak_cuda_allocated_bytes
    return remaining / gpu_total_memory_bytes


def project_embedding_time(
    *,
    batch_size: int,
    records_per_second: float,
    safe: bool,
    full_record_count: int = FULL_DATASET_RECORD_COUNT,
) -> EmbeddingTimeProjection:
    if records_per_second <= 0.0:
        return EmbeddingTimeProjection(
            batch_size=batch_size,
            records_per_second=records_per_second,
            projected_seconds=0.0,
            projected_hours=0.0,
            projected_days=0.0,
            safe=safe,
        )
    projected_seconds = full_record_count / records_per_second
    projected_hours = projected_seconds / 3600.0
    projected_days = projected_hours / 24.0
    return EmbeddingTimeProjection(
        batch_size=batch_size,
        records_per_second=records_per_second,
        projected_seconds=projected_seconds,
        projected_hours=projected_hours,
        projected_days=projected_days,
        safe=safe,
    )


def collect_token_budget_violations(
    token_counts: Sequence[int],
    *,
    token_budget: int,
) -> tuple[str, ...]:
    if token_budget <= 0:
        msg = "token_budget must be > 0"
        raise ValueError(msg)
    violations: list[str] = []
    for index, count in enumerate(token_counts):
        if count > token_budget:
            violations.append(
                f"record {index} re-encodes to {count} tokens (budget {token_budget})"
            )
    return tuple(violations)


def verify_bounded_token_budget(
    token_counts: Sequence[int],
    *,
    token_budget: int,
) -> None:
    violations = collect_token_budget_violations(token_counts, token_budget=token_budget)
    if violations:
        msg = "; ".join(violations)
        raise ValueError(msg)
