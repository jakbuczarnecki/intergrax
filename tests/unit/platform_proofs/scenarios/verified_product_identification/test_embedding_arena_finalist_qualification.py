"""Unit tests for controlled embedding finalist qualification."""

from __future__ import annotations

import ast
from pathlib import Path

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
from platform_proofs.scenarios.verified_product_identification.arena.composition.finalist_qualification_policy import (
    FINALIST_BGE_QWEN_QUALIFICATION_SELECTION,
    map_finalist_gate_to_decision,
    resolve_finalist_qualification_selection,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidate_selection import (
    EmbeddingArenaCandidateSelection,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    EmbeddingArenaStageStatus,
    EmbeddingArenaVerdict,
    FinalistQualificationGate,
    SpeedupBand,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    FinalistQualificationSelectionError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.finalist_qualification import (
    FinalistQualificationSelection,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    CandidateArenaResult,
    CandidateStageSnapshot,
    QualityDeltaMetrics,
    QueryLatencySnapshot,
    RetrievalQualityMetrics,
    SpeedupEstimate,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.finalist_qualification import (
    classify_finalist_qualification_gate,
)

pytestmark = pytest.mark.unit

_NEUTRAL_BASELINE_ID = "model-a"
_NEUTRAL_CHALLENGER_ID = "model-b"
_NEUTRAL_SELECTION = FinalistQualificationSelection(
    baseline_candidate_id=_NEUTRAL_BASELINE_ID,
    challenger_candidate_id=_NEUTRAL_CHALLENGER_ID,
)


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


def test_bge_qwen_composition_preset_roles() -> None:
    qualification = resolve_finalist_qualification_selection(
        FINALIST_BGE_QWEN_CANDIDATE_SELECTION
    )
    assert qualification is FINALIST_BGE_QWEN_QUALIFICATION_SELECTION
    assert qualification.baseline_candidate_id == BASELINE_CANDIDATE_ID
    assert qualification.challenger_candidate_id == QWEN_CANDIDATE_ID


def test_qualification_selection_rejects_same_baseline_and_challenger() -> None:
    with pytest.raises(ValueError, match="must differ"):
        FinalistQualificationSelection(
            baseline_candidate_id="model-a",
            challenger_candidate_id="model-a",
        )


def test_classify_baseline_clear_win() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.95,
        throughput=13.0,
        query_p50=0.015,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.90,
        throughput=9.0,
        query_p50=0.050,
        speedup=9.0 / 13.0,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.BASELINE_CLEAR_WIN


def test_classify_challenger_clear_win() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.050,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.92,
        throughput=20.0,
        query_p50=0.020,
        speedup=2.0,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.CHALLENGER_CLEAR_WIN


def test_classify_ambiguous() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.91,
        throughput=10.0,
        query_p50=0.030,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.90,
        throughput=10.5,
        query_p50=0.031,
        speedup=1.05,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.AMBIGUOUS


def test_classify_runtime_rejected_baseline() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.030,
        is_baseline=True,
        runtime_rejected=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.030,
        speedup=1.0,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.RUNTIME_REJECTED


def test_classify_runtime_rejected_challenger() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.030,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.030,
        speedup=1.0,
        runtime_rejected=True,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.RUNTIME_REJECTED


def test_classify_quality_regression() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.95,
        recall_at_10=0.95,
        mrr_at_10=0.95,
        throughput=10.0,
        query_p50=0.030,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.95,
        recall_at_10=0.80,
        mrr_at_10=0.80,
        throughput=20.0,
        query_p50=0.010,
        speedup=2.0,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.QUALITY_REGRESSION


def test_classify_more_evidence_required() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.85,
        throughput=10.0,
        query_p50=0.040,
        is_baseline=True,
    )
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.86,
        throughput=12.0,
        query_p50=0.034,
        speedup=1.2,
    )
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        _NEUTRAL_SELECTION,
    )
    assert gate is FinalistQualificationGate.MORE_EVIDENCE_REQUIRED


def test_missing_baseline_fails_closed() -> None:
    challenger = _candidate_result(
        candidate_id=_NEUTRAL_CHALLENGER_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.030,
        speedup=1.0,
    )
    with pytest.raises(FinalistQualificationSelectionError, match="baseline"):
        classify_finalist_qualification_gate((challenger,), _NEUTRAL_SELECTION)


def test_missing_challenger_fails_closed() -> None:
    baseline = _candidate_result(
        candidate_id=_NEUTRAL_BASELINE_ID,
        recall_at_1=0.90,
        throughput=10.0,
        query_p50=0.030,
        is_baseline=True,
    )
    with pytest.raises(FinalistQualificationSelectionError, match="challenger"):
        classify_finalist_qualification_gate((baseline,), _NEUTRAL_SELECTION)


def test_bge_qwen_composition_gate_mapping() -> None:
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
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        FINALIST_BGE_QWEN_QUALIFICATION_SELECTION,
    )
    assert gate is FinalistQualificationGate.BASELINE_CLEAR_WIN
    assert (
        map_finalist_gate_to_decision(gate, FINALIST_BGE_QWEN_QUALIFICATION_SELECTION)
        is EmbeddingArenaDecision.KEEP_BGE_M3
    )


def test_bge_qwen_challenger_win_maps_to_promote_qwen() -> None:
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
    gate, _ = classify_finalist_qualification_gate(
        (baseline, challenger),
        FINALIST_BGE_QWEN_QUALIFICATION_SELECTION,
    )
    assert gate is FinalistQualificationGate.CHALLENGER_CLEAR_WIN
    assert (
        map_finalist_gate_to_decision(gate, FINALIST_BGE_QWEN_QUALIFICATION_SELECTION)
        is EmbeddingArenaDecision.PROMOTE_QWEN3_0_6B_CANDIDATE
    )


def test_nano_and_safe_profiles_unchanged() -> None:
    nano = resolve_execution_budget("nano-local-gpu")
    safe = resolve_execution_budget("safe-local-gpu")
    assert nano.finalist_qualification_mode is False
    assert safe.finalist_qualification_mode is False
    assert nano.screening_mode is True
    assert safe.screening_mode is True


def test_finalist_qualification_evaluator_has_no_composition_candidate_imports() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    module_path = (
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/arena/evaluation/finalist_qualification.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_modules = {
        "platform_proofs.scenarios.verified_product_identification.arena.composition.candidates",
        "platform_proofs.scenarios.verified_product_identification.arena.composition.candidate_selection",
        "platform_proofs.scenarios.verified_product_identification.arena.composition.finalist_qualification_policy",
    }
    forbidden_names = {"BASELINE_CANDIDATE_ID", "QWEN_CANDIDATE_ID"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module in forbidden_modules:
                violations.append(f"import from {node.module}")
            for alias in node.names:
                if alias.name in forbidden_names:
                    violations.append(f"import name {alias.name}")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            violations.append(f"reference to {node.id}")
    assert violations == []


def _candidate_result(
    *,
    candidate_id: str,
    recall_at_1: float,
    throughput: float,
    query_p50: float,
    recall_at_10: float | None = None,
    mrr_at_10: float | None = None,
    speedup: float | None = None,
    is_baseline: bool = False,
    runtime_rejected: bool = False,
) -> CandidateArenaResult:
    recall_10 = recall_at_10 if recall_at_10 is not None else 1.0
    mrr_10 = mrr_at_10 if mrr_at_10 is not None else recall_at_1
    quality = RetrievalQualityMetrics(
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_1,
        recall_at_10=recall_10,
        mrr_at_10=mrr_10,
        ndcg_at_10=recall_at_1,
        query_count=5,
    )
    stage_status = (
        EmbeddingArenaStageStatus.FAILED_RUNTIME
        if runtime_rejected
        else EmbeddingArenaStageStatus.PASS
    )
    stage_c = CandidateStageSnapshot(
        stage_name="stage_c",
        record_count=100,
        status=stage_status,
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
        EmbeddingArenaVerdict.REJECTED_RUNTIME
        if runtime_rejected
        else (
            EmbeddingArenaVerdict.BASELINE
            if is_baseline
            else EmbeddingArenaVerdict.QUALIFIED
        )
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
