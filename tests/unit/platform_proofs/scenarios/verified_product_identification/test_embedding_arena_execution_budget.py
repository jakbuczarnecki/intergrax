"""Unit tests for embedding arena execution budgets and profiles."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    BASELINE_CANDIDATE_ID,
    build_default_arena_candidates,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.execution_profiles import (
    NANO_LOCAL_GPU_ARENA_EXECUTION_BUDGET,
    NANO_LOCAL_GPU_ARENA_PROFILE_ID,
    SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
    SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID,
    STANDARD_ARENA_EXECUTION_BUDGET,
    resolve_execution_budget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_budget import (
    EmbeddingArenaExecutionBudget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_environment import (
    ArenaAcceleratorRequirement,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.finalist_selection import (
    StageBCandidateEvidence,
    select_stage_c_finalist_ids,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)

pytestmark = pytest.mark.unit


def test_safe_local_gpu_profile_sizes() -> None:
    budget = SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET
    assert budget.accelerator_requirement is ArenaAcceleratorRequirement.CUDA
    assert budget.stage_a_records == 20
    assert budget.stage_b_records == 50
    assert budget.stage_c_records == 100
    assert budget.max_stage_c_finalists == 2
    assert budget.default_batch_size == 16
    assert budget.fallback_batch_size == 8
    assert budget.candidate_timeout_seconds == 900.0
    assert budget.batch_sweep_sizes == ()
    assert budget.isolate_candidates is True
    assert budget.screening_mode is True


def test_standard_profile_preserves_legacy_sizes() -> None:
    budget = STANDARD_ARENA_EXECUTION_BUDGET
    assert budget.accelerator_requirement is ArenaAcceleratorRequirement.ANY
    assert budget.stage_a_records == 100
    assert budget.stage_b_records == 500
    assert budget.stage_c_records == 1000
    assert budget.max_stage_c_finalists == 3
    assert budget.batch_sweep_sizes == (8, 16, 32, 64)


def test_invalid_profile_rejected() -> None:
    with pytest.raises(ValueError, match="unknown arena profile"):
        resolve_execution_budget("does-not-exist")


def test_execution_budget_validation_rejects_bad_stage_order() -> None:
    with pytest.raises(ValueError, match="stage_a_records must be <= stage_b_records"):
        EmbeddingArenaExecutionBudget(
            profile_id="bad",
            accelerator_requirement=ArenaAcceleratorRequirement.ANY,
            stage_a_records=50,
            stage_b_records=20,
            stage_c_records=100,
            max_stage_c_finalists=2,
            candidate_timeout_seconds=60.0,
            default_batch_size=16,
            fallback_batch_size=8,
            batch_sweep_sizes=(),
            isolate_candidates=False,
            screening_mode=False,
            max_vram_bytes=None,
            query_latency_repetitions=3,
            query_latency_query_count=3,
            max_total_wall_time_seconds=None,
            run_long_input_quality_benchmark=True,
            include_full_build_estimate=True,
            include_query_latency_benchmark=True,
            suppress_keep_baseline_decision=False,
            screening_evidence_label=None,
        )


def test_micro_batch_policy_is_primary_then_fallback_only() -> None:
    budget = SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET
    candidates = build_default_arena_candidates(include_e5_control=False)
    bge = next(item for item in candidates if item.candidate_id == BASELINE_CANDIDATE_ID)
    qwen = next(item for item in candidates if item.candidate_id == "qwen3-0.6b")

    assert budget.batch_sizes_for_candidate(
        fixed_provider_batch_size=bge.fixed_provider_batch_size,
    ) == (16,)
    assert budget.batch_sizes_for_candidate(
        fixed_provider_batch_size=qwen.fixed_provider_batch_size,
    ) == (16, 8)


def test_micro_profile_caps_stage_c_finalists_at_two() -> None:
    baseline_throughput = 10.0
    evidence = tuple(
        StageBCandidateEvidence(
            candidate_id=candidate_id,
            is_baseline=candidate_id == BASELINE_CANDIDATE_ID,
            license_eligible=True,
            runtime_ok=True,
            throughput_records_per_second=throughput,
            quality_metrics=RetrievalQualityMetrics(
                recall_at_1=recall,
                recall_at_5=recall,
                recall_at_10=recall,
                mrr_at_10=recall,
                ndcg_at_10=recall,
                query_count=5,
            ),
        )
        for candidate_id, recall, throughput in (
            (BASELINE_CANDIDATE_ID, 0.90, baseline_throughput),
            ("qwen3-0.6b", 0.95, baseline_throughput * 2.0),
            ("nomic-v2-moe", 0.94, baseline_throughput * 3.0),
        )
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET.max_stage_c_finalists,
    )
    assert len(finalists) == 2


def test_resolve_safe_local_gpu_profile_by_id() -> None:
    budget = resolve_execution_budget(SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID)
    assert budget.profile_id == SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID


def test_nano_local_gpu_profile_sizes() -> None:
    budget = NANO_LOCAL_GPU_ARENA_EXECUTION_BUDGET
    assert budget.accelerator_requirement is ArenaAcceleratorRequirement.CUDA
    assert budget.stage_a_records == 5
    assert budget.stage_b_records == 10
    assert budget.stage_c_records == 20
    assert budget.max_stage_c_finalists == 2
    assert budget.default_batch_size == 8
    assert budget.fallback_batch_size == 4
    assert budget.candidate_timeout_seconds == 180.0
    assert budget.batch_sweep_sizes == ()
    assert budget.isolate_candidates is True
    assert budget.screening_mode is True
    assert budget.uses_batch_sweep is False
    assert budget.max_total_wall_time_seconds == 20.0 * 60.0
    assert budget.run_long_input_quality_benchmark is False
    assert budget.include_full_build_estimate is False
    assert budget.query_latency_repetitions == 1
    assert budget.query_latency_query_count == 2
    assert budget.suppress_keep_baseline_decision is True
    assert budget.screening_evidence_label == "NANO SCREENING ONLY"


def test_resolve_nano_local_gpu_profile_by_id() -> None:
    budget = resolve_execution_budget(NANO_LOCAL_GPU_ARENA_PROFILE_ID)
    assert budget.profile_id == NANO_LOCAL_GPU_ARENA_PROFILE_ID


def test_nano_batch_policy_is_primary_then_fallback_only() -> None:
    budget = NANO_LOCAL_GPU_ARENA_EXECUTION_BUDGET
    candidates = build_default_arena_candidates(include_e5_control=False)
    bge = next(item for item in candidates if item.candidate_id == BASELINE_CANDIDATE_ID)
    qwen = next(item for item in candidates if item.candidate_id == "qwen3-0.6b")

    assert budget.batch_sizes_for_candidate(
        fixed_provider_batch_size=bge.fixed_provider_batch_size,
    ) == (16,)
    assert budget.batch_sizes_for_candidate(
        fixed_provider_batch_size=qwen.fixed_provider_batch_size,
    ) == (8, 4)


def test_nano_profile_caps_stage_c_finalists_at_two() -> None:
    baseline_throughput = 10.0
    evidence = tuple(
        StageBCandidateEvidence(
            candidate_id=candidate_id,
            is_baseline=candidate_id == BASELINE_CANDIDATE_ID,
            license_eligible=True,
            runtime_ok=True,
            throughput_records_per_second=throughput,
            quality_metrics=RetrievalQualityMetrics(
                recall_at_1=recall,
                recall_at_5=recall,
                recall_at_10=recall,
                mrr_at_10=recall,
                ndcg_at_10=recall,
                query_count=5,
            ),
        )
        for candidate_id, recall, throughput in (
            (BASELINE_CANDIDATE_ID, 0.90, baseline_throughput),
            ("qwen3-0.6b", 0.95, baseline_throughput * 2.0),
            ("nomic-v2-moe", 0.94, baseline_throughput * 3.0),
        )
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=NANO_LOCAL_GPU_ARENA_EXECUTION_BUDGET.max_stage_c_finalists,
    )
    assert len(finalists) == 2


def test_safe_local_gpu_profile_unchanged() -> None:
    budget = resolve_execution_budget(SAFE_LOCAL_GPU_MICRO_ARENA_PROFILE_ID)
    assert budget.stage_a_records == 20
    assert budget.stage_b_records == 50
    assert budget.stage_c_records == 100
    assert budget.candidate_timeout_seconds == 900.0
    assert budget.default_batch_size == 16
    assert budget.fallback_batch_size == 8
    assert budget.include_full_build_estimate is True
    assert budget.run_long_input_quality_benchmark is True
