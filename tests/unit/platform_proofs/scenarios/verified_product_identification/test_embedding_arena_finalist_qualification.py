"""Unit tests for controlled embedding finalist qualification."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.arena.composition.candidate_selection import (
    FINALIST_BGE_QWEN_CANDIDATE_SELECTION,
    resolve_arena_candidates,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    BASELINE_CANDIDATE_ID,
    NOMIC_CANDIDATE_ID,
    QWEN_CANDIDATE_ID,
    build_default_arena_candidates,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.execution_profiles import (
    FINALIST_LOCAL_GPU_200_ARENA_EXECUTION_BUDGET,
    FINALIST_LOCAL_GPU_ARENA_EXECUTION_BUDGET,
    FINALIST_LOCAL_GPU_ARENA_PROFILE_ID,
    NANO_LOCAL_GPU_ARENA_EXECUTION_BUDGET,
    SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
    resolve_execution_budget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidate_selection import (
    EmbeddingArenaCandidateSelection,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    EmbeddingArenaStageStatus,
    EmbeddingArenaVerdict,
    FinalistQualificationGate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    CandidateArenaResult,
    CandidateStageSnapshot,
    QualityDeltaMetrics,
    RetrievalQualityMetrics,
    SpeedupEstimate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    SpeedupBand,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.finalist_qualification import (
    classify_finalist_qualification_gate,
    map_finalist_gate_to_decision,
)

pytestmark = pytest.mark.unit


def test_finalist_local_gpu_profile_sizes() -> None:
    budget = FINALIST_LOCAL_GPU_ARENA_EXECUTION_BUDGET
    assert budget.profile_id == FINALIST_LOCAL_GPU_ARENA_PROFILE_ID
    assert budget.stage_a_records == 100
    assert budget.stage_b_records == 100
    assert budget.stage_c_records == 100
    assert budget.max_stage_c_finalists == 2
    assert budget.candidate_timeout_seconds == 300.0
    assert budget.batch_sweep_sizes == ()
    assert budget.isolate_candidates is True
    assert budget.screening_mode is False
    assert budget.finalist_qualification_mode is True
    assert budget.default_candidate_selection == FINALIST_BGE_QWEN_CANDIDATE_SELECTION
    assert budget.max_total_wall_time_seconds == 15.0 * 60.0


def test_finalist_local_gpu_200_profile_sizes() -> None:
    budget = FINALIST_LOCAL_GPU_200_ARENA_EXECUTION_BUDGET
    assert budget.stage_c_records == 200
    assert budget.stage_a_records == 200
    assert budget.stage_b_records == 200


def test_finalist_profile_fixed_batches_without_sweep() -> None:
    budget = FINALIST_LOCAL_GPU_ARENA_EXECUTION_BUDGET
    candidates = build_default_arena_candidates(include_e5_control=False)
    bge = next(item for item in candidates if item.candidate_id == BASELINE_CANDIDATE_ID)
    qwen = next(item for item in candidates if item.candidate_id == QWEN_CANDIDATE_ID)

    assert budget.batch_sizes_for_candidate(
        candidate_id=bge.candidate_id,
        fixed_provider_batch_size=bge.fixed_provider_batch_size,
    ) == (16,)
    assert budget.batch_sizes_for_candidate(
        candidate_id=qwen.candidate_id,
        fixed_provider_batch_size=qwen.fixed_provider_batch_size,
    ) == (8,)


def test_finalist_selection_includes_only_configured_candidates() -> None:
    candidates = resolve_arena_candidates(selection=FINALIST_BGE_QWEN_CANDIDATE_SELECTION)
    ids = {candidate.candidate_id for candidate in candidates}
    assert ids == {BASELINE_CANDIDATE_ID, QWEN_CANDIDATE_ID}
    assert NOMIC_CANDIDATE_ID not in ids


def test_finalist_selection_preserves_deterministic_order() -> None:
    candidates = resolve_arena_candidates(selection=FINALIST_BGE_QWEN_CANDIDATE_SELECTION)
    assert tuple(candidate.candidate_id for candidate in candidates) == (
        BASELINE_CANDIDATE_ID,
        QWEN_CANDIDATE_ID,
    )


def test_unknown_candidate_in_selection_rejected() -> None:
    selection = EmbeddingArenaCandidateSelection(candidate_ids=("missing-id",))
    with pytest.raises(ValueError, match="unknown arena candidate id"):
        resolve_arena_candidates(selection=selection)


def test_classify_bge_clear_win() -> None:
    baseline = _candidate_result(
        candidate_id=BASELINE_CANDIDATE_ID,
        recall_at_1=0.95,
        throughput=13.0,
        query_p50=0.015,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=QWEN_CANDIDATE_ID,
        recall_at_1=0.90,
        throughput=9.0,
        query_p50=0.050,
        speedup=9.0 / 13.0,
    )
    gate, _ = classify_finalist_qualification_gate((baseline, challenger))
    assert gate is FinalistQualificationGate.BGE_CLEAR_WIN
    assert map_finalist_gate_to_decision(gate) is EmbeddingArenaDecision.KEEP_BGE_M3


def test_classify_qwen_clear_win() -> None:
    baseline = _candidate_result(
        candidate_id=BASELINE_CANDIDATE_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.050,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=QWEN_CANDIDATE_ID,
        recall_at_1=0.92,
        throughput=20.0,
        query_p50=0.020,
        speedup=2.0,
    )
    gate, _ = classify_finalist_qualification_gate((baseline, challenger))
    assert gate is FinalistQualificationGate.QWEN_CLEAR_WIN
    assert (
        map_finalist_gate_to_decision(gate)
        is EmbeddingArenaDecision.PROMOTE_QWEN3_0_6B_CANDIDATE
    )


def test_classify_ambiguous() -> None:
    baseline = _candidate_result(
        candidate_id=BASELINE_CANDIDATE_ID,
        recall_at_1=0.91,
        throughput=10.0,
        query_p50=0.030,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=QWEN_CANDIDATE_ID,
        recall_at_1=0.90,
        throughput=10.5,
        query_p50=0.031,
        speedup=1.05,
    )
    gate, _ = classify_finalist_qualification_gate((baseline, challenger))
    assert gate is FinalistQualificationGate.AMBIGUOUS


def test_nano_and_safe_profiles_unchanged() -> None:
    nano = resolve_execution_budget("nano-local-gpu")
    safe = resolve_execution_budget("safe-local-gpu")
    assert nano.finalist_qualification_mode is False
    assert safe.finalist_qualification_mode is False
    assert nano.screening_mode is True
    assert safe.screening_mode is True


def _candidate_result(
    *,
    candidate_id: str,
    recall_at_1: float,
    throughput: float,
    query_p50: float,
    speedup: float | None = None,
    is_baseline: bool = False,
) -> CandidateArenaResult:
    quality = RetrievalQualityMetrics(
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_1,
        recall_at_10=1.0,
        mrr_at_10=recall_at_1,
        ndcg_at_10=recall_at_1,
        query_count=5,
    )
    from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
        QueryLatencySnapshot,
    )

    stage_c = CandidateStageSnapshot(
        stage_name="stage_c",
        record_count=100,
        status=EmbeddingArenaStageStatus.PASS,
        selected_provider_batch_size=16 if is_baseline else 8,
        warmup_timing=None,
        microbenchmark_results=(),
        throughput_records_per_second=throughput,
        peak_vram_bytes=1_000_000,
        output_dimension=1024,
        detail=None,
    )
    speedup_estimate = None
    if speedup is not None:
        speedup_estimate = SpeedupEstimate(
            speedup_vs_baseline=speedup,
            speedup_band=SpeedupBand.BAND_2_TO_4X,
            hours_saved_vs_baseline=None,
        )
    verdict = (
        EmbeddingArenaVerdict.BASELINE
        if is_baseline
        else EmbeddingArenaVerdict.QUALIFIED
    )
    return CandidateArenaResult(
        candidate_id=candidate_id,
        verdict=verdict,
        runtime_metadata=None,
        truncation_profile=None,
        stage_a=None,
        stage_b=None,
        stage_c=stage_c,
        quality_metrics=quality,
        long_input_quality_metrics=None,
        quality_delta_vs_baseline=(
            None
            if is_baseline
            else QualityDeltaMetrics(
                delta_recall_at_1=0.0,
                delta_recall_at_5=0.0,
                delta_recall_at_10=0.0,
                delta_mrr_at_10=0.0,
                delta_ndcg_at_10=0.0,
            )
        ),
        query_latency=QueryLatencySnapshot(
            single_query_p50_seconds=query_p50,
            single_query_p95_seconds=query_p50 * 1.2,
            small_batch_records_per_second=throughput,
        ),
        artifact_size_estimate=None,
        full_build_estimate=None,
        speedup_estimate=speedup_estimate,
        warnings=(),
        screening_outcome=None,
    )
