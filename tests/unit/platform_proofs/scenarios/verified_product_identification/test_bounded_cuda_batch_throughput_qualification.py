"""Unit tests for bounded CUDA batch throughput qualification helpers."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    BATCH_32_MIN_IMPROVEMENT_FRACTION,
    BOUNDED_CUDA_THROUGHPUT_QUALIFICATION_CLOSEOUT,
    CLOSEOUT_TASK_ID,
    EFFECTIVE_PROVIDER_TOKEN_CEILING,
    MIN_VRAM_HEADROOM_FRACTION,
    BatchVariantMeasurement,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.metrics import (
    compute_throughput_metrics,
    compute_token_profile,
    compute_vram_headroom_fraction,
    project_embedding_time,
    verify_bounded_token_budget,
    verify_effective_provider_token_ceiling,
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
    safety_detail: str = "test",
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
        safety_detail=safety_detail,
    )


def _frozen_closeout_measurement(
    *,
    batch_size: int,
    records_per_second: float,
    previous_smaller: BatchVariantMeasurement | None,
) -> BatchVariantMeasurement:
    measurement = _measurement(
        batch_size=batch_size,
        records_per_second=records_per_second,
        peak_allocated=2_000,
    )
    safe, safety_detail = evaluate_batch_safety(
        measurement,
        previous_smaller=previous_smaller,
    )
    return _measurement(
        batch_size=batch_size,
        records_per_second=records_per_second,
        peak_allocated=2_000,
        safe=safe,
        safety_detail=safety_detail,
    )


def _frozen_closeout_measurements() -> tuple[BatchVariantMeasurement, ...]:
    batch_1 = _frozen_closeout_measurement(
        batch_size=1,
        records_per_second=23.42,
        previous_smaller=None,
    )
    batch_4 = _frozen_closeout_measurement(
        batch_size=4,
        records_per_second=22.36,
        previous_smaller=batch_1,
    )
    batch_8 = _frozen_closeout_measurement(
        batch_size=8,
        records_per_second=19.25,
        previous_smaller=batch_4,
    )
    batch_16 = _frozen_closeout_measurement(
        batch_size=16,
        records_per_second=19.31,
        previous_smaller=batch_8,
    )
    return (batch_1, batch_4, batch_8, batch_16)


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


def test_verify_effective_provider_token_ceiling_allows_boundary() -> None:
    verify_effective_provider_token_ceiling((100, EFFECTIVE_PROVIDER_TOKEN_CEILING))


def test_verify_effective_provider_token_ceiling_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="re-encodes to 771 tokens"):
        verify_effective_provider_token_ceiling((771,))


def test_frozen_bounded_cuda_throughput_qualification_closeout() -> None:
    closeout = BOUNDED_CUDA_THROUGHPUT_QUALIFICATION_CLOSEOUT

    assert closeout.closeout_task_id == CLOSEOUT_TASK_ID
    assert closeout.qualified_production_batch_size == 1
    assert closeout.qualified_records_per_second == pytest.approx(23.42)
    assert closeout.projected_embedding_only_hours == pytest.approx(44.7)
    assert closeout.projection_only is True
    assert closeout.optional_batch_32_executed is False

    evidence_by_batch = {item.batch_size: item for item in closeout.batch_evidence}

    assert evidence_by_batch[1].safe is True
    assert evidence_by_batch[1].selected is True
    assert evidence_by_batch[1].records_per_second == pytest.approx(23.42)

    assert evidence_by_batch[4].safe is False
    assert evidence_by_batch[4].selected is False
    assert evidence_by_batch[4].records_per_second == pytest.approx(22.36)
    assert "batch 1" in evidence_by_batch[4].notes

    assert evidence_by_batch[8].safe is False
    assert evidence_by_batch[8].selected is False
    assert evidence_by_batch[8].records_per_second == pytest.approx(19.25)
    assert "batch 4" in evidence_by_batch[8].notes

    assert evidence_by_batch[16].safe is True
    assert evidence_by_batch[16].selected is False
    assert evidence_by_batch[16].records_per_second == pytest.approx(19.31)

    assert evidence_by_batch[32].safe is False
    assert evidence_by_batch[32].selected is False
    assert evidence_by_batch[32].records_per_second is None
    assert "gating" in evidence_by_batch[32].notes

    selected_batches = [item.batch_size for item in closeout.batch_evidence if item.selected]
    assert selected_batches == [1]


def test_frozen_closeout_matches_evaluate_batch_safety_semantics() -> None:
    closeout = BOUNDED_CUDA_THROUGHPUT_QUALIFICATION_CLOSEOUT
    evidence_by_batch = {item.batch_size: item for item in closeout.batch_evidence}
    measurements = _frozen_closeout_measurements()

    previous: BatchVariantMeasurement | None = None
    for measurement in measurements:
        safe, _detail = evaluate_batch_safety(
            measurement,
            previous_smaller=previous,
        )
        assert safe is evidence_by_batch[measurement.batch_size].safe
        previous = measurement


def test_frozen_closeout_select_production_batch_candidate_returns_batch_1() -> None:
    measurements = _frozen_closeout_measurements()

    selection = select_production_batch_candidate(measurements)

    assert selection.batch_size == 1


def test_no_forbidden_contract_patterns_in_bounded_cuda_batch_modules() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[5]
    bounded_root = (
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/qualification/bounded_cuda_batch"
    )
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "getattr",
        "setattr",
        "hasattr",
    )
    for module_path in sorted(bounded_root.glob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {module_path.name}"


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
