"""Microbenchmark candidate selection — provider-neutral."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    MicrobenchmarkCandidateResult,
    MicrobenchmarkCandidateStatus,
)


def select_best_provider_batch_size(
    candidates: tuple[MicrobenchmarkCandidateResult, ...],
    *,
    expected_dimension: int,
) -> tuple[int | None, str]:
    passing = [
        candidate
        for candidate in candidates
        if candidate.status is MicrobenchmarkCandidateStatus.PASS
        and candidate.records_per_second > 0.0
    ]
    if not passing:
        return None, "no stable microbenchmark candidate completed successfully"
    passing_sorted = sorted(passing, key=lambda item: item.records_per_second, reverse=True)
    best = passing_sorted[0]
    if len(passing_sorted) > 1:
        runner_up = passing_sorted[1]
        if best.peak_vram_bytes is not None and runner_up.peak_vram_bytes is not None:
            vram_headroom = best.peak_vram_bytes - runner_up.peak_vram_bytes
            throughput_delta = best.records_per_second - runner_up.records_per_second
            if vram_headroom > 0 and throughput_delta / best.records_per_second < 0.05:
                return (
                    runner_up.provider_batch_size,
                    (
                        "selected lower batch size "
                        f"{runner_up.provider_batch_size} over {best.provider_batch_size} "
                        "for similar throughput with better VRAM headroom"
                    ),
                )
    return (
        best.provider_batch_size,
        (
            f"highest stable throughput among tested configurations "
            f"({best.records_per_second:.2f} records/s at provider batch {best.provider_batch_size})"
        ),
    )
