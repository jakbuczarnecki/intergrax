"""Unit tests for VPI embedding qualification contracts and helpers."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.qualification.batch_selection import (
    select_best_provider_batch_size,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bottleneck import (
    analyze_bottleneck,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    MicrobenchmarkCandidateResult,
    MicrobenchmarkCandidateStatus,
)
from platform_proofs.scenarios.verified_product_identification.qualification.duration_estimate import (
    estimate_full_build_duration,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.microbenchmark import (
    _is_cuda_oom,
)
from platform_proofs.scenarios.verified_product_identification.qualification.text_length_profile import (
    profile_text_lengths,
)

pytestmark = pytest.mark.unit


def test_text_length_profile_percentiles() -> None:
    profile = profile_text_lengths(("a", "bbb", "ccccc"))

    assert profile.character_min == 1
    assert profile.character_max == 5
    assert profile.character_mean == pytest.approx(3.0)


def test_select_best_provider_batch_size_prefers_throughput() -> None:
    candidates = (
        MicrobenchmarkCandidateResult(
            provider_batch_size=16,
            record_count=128,
            embed_elapsed_seconds=10.0,
            records_per_second=12.8,
            status=MicrobenchmarkCandidateStatus.PASS,
            peak_vram_bytes=1_000,
            detail=None,
        ),
        MicrobenchmarkCandidateResult(
            provider_batch_size=32,
            record_count=128,
            embed_elapsed_seconds=5.0,
            records_per_second=25.6,
            status=MicrobenchmarkCandidateStatus.PASS,
            peak_vram_bytes=2_000,
            detail=None,
        ),
        MicrobenchmarkCandidateResult(
            provider_batch_size=64,
            record_count=128,
            embed_elapsed_seconds=0.0,
            records_per_second=0.0,
            status=MicrobenchmarkCandidateStatus.FAILED_OOM,
            peak_vram_bytes=None,
            detail="oom",
        ),
    )

    selected, rationale = select_best_provider_batch_size(
        candidates,
        expected_dimension=1024,
    )

    assert selected == 32
    assert "throughput" in rationale


def test_duration_estimate_scales_linearly() -> None:
    estimate = estimate_full_build_duration(
        record_count=1_000,
        steady_records_per_second=10.0,
        derive_seconds_per_record=0.001,
        artifact_write_seconds_per_record=0.002,
        throughput_source="unit_test",
    )

    assert estimate.estimated_embedding_seconds == pytest.approx(100.0)
    assert estimate.estimated_total_seconds == pytest.approx(103.0)


def test_bottleneck_embedding_dominant() -> None:
    breakdown = analyze_bottleneck(
        derive_seconds=1.0,
        embedding_seconds=90.0,
        artifact_write_seconds=1.0,
    )

    assert breakdown.dominant_stage == "embedding"
    assert breakdown.embedding_share == pytest.approx(0.978, rel=1e-3)
    assert "embedding dominates" in breakdown.parallelization_recommendation


def test_cuda_oom_classification_message() -> None:
    assert _is_cuda_oom(RuntimeError("CUDA out of memory")) is True
    assert _is_cuda_oom(RuntimeError("other failure")) is False
