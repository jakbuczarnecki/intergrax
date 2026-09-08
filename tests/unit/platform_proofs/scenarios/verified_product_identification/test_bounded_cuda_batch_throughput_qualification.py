"""Unit tests for bounded CUDA batch throughput qualification helpers."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    BATCH_32_MIN_IMPROVEMENT_FRACTION,
    MIN_VRAM_HEADROOM_FRACTION,
    BatchVariantMeasurement,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.metrics import (
    compute_throughput_metrics,
    compute_token_profile,
    compute_vram_headroom_fraction,
    project_embedding_time,
    verify_bounded_token_budget,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.reporting import (
    bounded_cuda_report_to_json,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.selection import (
    evaluate_batch_safety,
    select_production_batch_candidate,
    should_run_optional_batch_32,
)

pytestmark = pytest.mark.unit


def _measurement(
    *,
    batch_size: int,
    records_per_second: float,
    peak_allocated: int,
    total_vram: int = 10_000,
    safe: bool = True,
    cuda_oom: bool = False,
) -> BatchVariantMeasurement:
    headroom = compute_vram_headroom_fraction(
        gpu_total_memory_bytes=total_vram,
        peak_cuda_allocated_bytes=peak_allocated,
    )
    return BatchVariantMeasurement(
        batch_size=batch_size,
        record_count=32,
        total_bounded_input_tokens=1000,
        average_tokens_per_record=31.25,
        p50_tokens=30.0,
        p95_tokens=40.0,
        max_tokens=45,
        wall_clock_embedding_seconds=1.0,
        records_per_second=records_per_second,
        tokens_per_second=1000.0,
        average_milliseconds_per_record=31.25,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_allocated,
        gpu_total_memory_bytes=total_vram,
        gpu_free_memory_bytes_before=total_vram - peak_allocated,
        gpu_free_memory_bytes_after=total_vram - peak_allocated,
        cuda_oom=cuda_oom,
        vram_headroom_fraction=headroom,
        safe=safe,
        safety_detail="test",
    )


def test_compute_token_profile_percentiles() -> None:
    profile = compute_token_profile((10, 20, 30, 40))

    assert profile.total_tokens == 100
    assert profile.average_tokens_per_record == pytest.approx(25.0)
    assert profile.max_tokens == 40


def test_compute_throughput_metrics() -> None:
    records_per_second, tokens_per_second, average_ms = compute_throughput_metrics(
        record_count=32,
        total_tokens=640,
        wall_clock_seconds=2.0,
    )

    assert records_per_second == pytest.approx(16.0)
    assert tokens_per_second == pytest.approx(320.0)
    assert average_ms == pytest.approx(62.5)


def test_project_embedding_time_scales_linearly() -> None:
    projection = project_embedding_time(
        batch_size=16,
        records_per_second=10.0,
        safe=True,
        full_record_count=1_000,
    )

    assert projection.projected_seconds == pytest.approx(100.0)
    assert projection.projected_hours == pytest.approx(100.0 / 3600.0)


def test_verify_bounded_token_budget_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="re-encodes to 800 tokens"):
        verify_bounded_token_budget((100, 800), token_budget=768)


def test_select_production_batch_candidate_prefers_safe_highest_throughput() -> None:
    measurements = (
        _measurement(batch_size=1, records_per_second=5.0, peak_allocated=1_000),
        _measurement(batch_size=4, records_per_second=12.0, peak_allocated=2_000),
        _measurement(batch_size=8, records_per_second=15.0, peak_allocated=3_000),
        _measurement(
            batch_size=16,
            records_per_second=20.0,
            peak_allocated=9_500,
            safe=False,
        ),
    )

    selection = select_production_batch_candidate(measurements)

    assert selection.batch_size == 8
    assert "highest stable throughput" in selection.rationale


def test_memory_headroom_rule_marks_unsafe_variant() -> None:
    measurement = _measurement(
        batch_size=16,
        records_per_second=20.0,
        peak_allocated=9_000,
        total_vram=10_000,
    )

    safe, detail = evaluate_batch_safety(measurement, previous_smaller=None)

    assert safe is False
    assert "VRAM headroom" in detail


def test_optional_batch_32_gating_requires_material_improvement() -> None:
    batch_8 = _measurement(batch_size=8, records_per_second=10.0, peak_allocated=2_000)
    batch_16 = _measurement(batch_size=16, records_per_second=10.5, peak_allocated=3_000)

    decision = should_run_optional_batch_32(
        batch_8=batch_8,
        batch_16=batch_16,
        minimum_improvement_fraction=BATCH_32_MIN_IMPROVEMENT_FRACTION,
    )

    assert decision.executed is False
    assert "improvement" in decision.reason


def test_optional_batch_32_gating_allows_run_when_conditions_met() -> None:
    batch_8 = _measurement(batch_size=8, records_per_second=10.0, peak_allocated=2_000)
    batch_16 = _measurement(batch_size=16, records_per_second=12.0, peak_allocated=3_000)

    decision = should_run_optional_batch_32(
        batch_8=batch_8,
        batch_16=batch_16,
        minimum_headroom_fraction=MIN_VRAM_HEADROOM_FRACTION,
    )

    assert decision.executed is True


def test_bounded_cuda_report_json_serializes_enums() -> None:
    from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
        BoundedCudaBatchThroughputReport,
        CudaPreflightSnapshot,
        OptionalBatch32Decision,
        ProductionBatchSelection,
        QualificationRunStatus,
        TokenProfile,
    )

    report = BoundedCudaBatchThroughputReport(
        task_id="test",
        status=QualificationRunStatus.PASS,
        python_executable="python",
        preflight=CudaPreflightSnapshot(
            python_version="3.12",
            torch_version="2.0",
            cuda_runtime_version="12.0",
            cuda_available=True,
            gpu_name="gpu",
            gpu_total_memory_bytes=100,
            gpu_free_memory_bytes_before_load=80,
            gpu_used_memory_bytes_before_load=20,
            resource_precondition_fail_reason=None,
        ),
        provider="hf",
        model="BAAI/bge-m3",
        revision="rev",
        dimension=1024,
        model_load_count=1,
        policy_version="vpi-bge-m3-document-token-budget-768-v1",
        token_budget=768,
        dataset_path="dataset.parquet",
        record_count=32,
        selection_method="deterministic_first_n_real_offers",
        token_profile=TokenProfile(100, 3.0, 3.0, 4.0, 5),
        batch_measurements=(),
        optional_batch_32=OptionalBatch32Decision(False, "skipped"),
        production_batch_selection=ProductionBatchSelection(8, "winner"),
        projections=(),
        winner_projection=None,
        projection_only=True,
        oom_observed=False,
        system_instability=False,
        persistent_vram_growth=False,
        known_gaps=(),
    )

    payload = bounded_cuda_report_to_json(report)

    assert '"status": "PASS"' in payload
