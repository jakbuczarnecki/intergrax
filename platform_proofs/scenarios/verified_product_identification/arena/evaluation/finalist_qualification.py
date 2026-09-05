"""Finalist qualification decision gate — BGE vs challenger on controlled sample."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    BASELINE_CANDIDATE_ID,
    QWEN_CANDIDATE_ID,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    EmbeddingArenaStageStatus,
    EmbeddingArenaVerdict,
    FinalistQualificationGate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    CandidateArenaResult,
    QualityDeltaMetrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.verdict import (
    compute_quality_delta,
    quality_non_regression_gate,
)

_CLOSE_QUALITY_DELTA = 0.02
_MATERIAL_QUALITY_DROP = 0.02
_MATERIAL_SPEEDUP = 1.15
_STRONG_SPEEDUP = 1.5
_MATERIAL_LATENCY_RATIO = 0.75


def _runtime_rejected(result: CandidateArenaResult) -> bool:
    return result.verdict is EmbeddingArenaVerdict.REJECTED_RUNTIME or (
        result.stage_c is not None
        and result.stage_c.status
        in {
            EmbeddingArenaStageStatus.FAILED_RUNTIME,
            EmbeddingArenaStageStatus.FAILED_RUNTIME_BUDGET,
            EmbeddingArenaStageStatus.FAILED_OOM,
        }
    )


def _quality_metrics_ready(result: CandidateArenaResult) -> bool:
    return result.quality_metrics is not None and result.stage_c is not None


def classify_finalist_qualification_gate(
    results: tuple[CandidateArenaResult, ...],
) -> tuple[FinalistQualificationGate, str]:
    baseline = next(
        (item for item in results if item.candidate_id == BASELINE_CANDIDATE_ID),
        None,
    )
    challenger = next(
        (item for item in results if item.candidate_id == QWEN_CANDIDATE_ID),
        None,
    )
    if baseline is None or challenger is None:
        return (
            FinalistQualificationGate.RUNTIME_REJECTED,
            "missing baseline or qwen finalist evidence",
        )
    if _runtime_rejected(baseline) or _runtime_rejected(challenger):
        return (
            FinalistQualificationGate.RUNTIME_REJECTED,
            "one or more finalists failed runtime qualification",
        )
    if not _quality_metrics_ready(baseline) or not _quality_metrics_ready(challenger):
        return (
            FinalistQualificationGate.RUNTIME_REJECTED,
            "quality metrics unavailable for finalist comparison",
        )

    assert baseline.quality_metrics is not None
    assert challenger.quality_metrics is not None
    delta = compute_quality_delta(challenger.quality_metrics, baseline.quality_metrics)
    if not quality_non_regression_gate(delta):
        return (
            FinalistQualificationGate.QUALITY_REGRESSION,
            "challenger fails quality non-regression on recall@10 or mrr@10",
        )

    speedup = (
        challenger.speedup_estimate.speedup_vs_baseline
        if challenger.speedup_estimate is not None
        else 1.0
    )
    latency_ratio = 1.0
    if (
        baseline.query_latency is not None
        and challenger.query_latency is not None
        and baseline.query_latency.single_query_p50_seconds > 0.0
    ):
        latency_ratio = (
            challenger.query_latency.single_query_p50_seconds
            / baseline.query_latency.single_query_p50_seconds
        )

    if _is_ambiguous(delta, speedup=speedup):
        return (
            FinalistQualificationGate.AMBIGUOUS,
            "decision-critical quality metrics and throughput remain within sample-noise band",
        )
    if _is_qwen_clear_win(delta, speedup=speedup, latency_ratio=latency_ratio):
        return (
            FinalistQualificationGate.QWEN_CLEAR_WIN,
            "challenger matches or exceeds baseline on recall@1/mrr/ndcg with runtime advantage",
        )
    if _is_bge_clear_win(delta, speedup=speedup, latency_ratio=latency_ratio):
        return (
            FinalistQualificationGate.BGE_CLEAR_WIN,
            "baseline leads decision-critical quality without challenger runtime advantage",
        )
    return (
        FinalistQualificationGate.MORE_EVIDENCE_REQUIRED,
        "no clear finalist winner on combined quality and runtime evidence",
    )


def _is_ambiguous(delta: QualityDeltaMetrics, *, speedup: float) -> bool:
    key_deltas = (
        abs(delta.delta_recall_at_1),
        abs(delta.delta_mrr_at_10),
        abs(delta.delta_ndcg_at_10),
    )
    quality_close = all(value <= _CLOSE_QUALITY_DELTA for value in key_deltas)
    throughput_close = (1.0 / _MATERIAL_SPEEDUP) <= speedup <= _MATERIAL_SPEEDUP
    return quality_close and throughput_close


def _is_qwen_clear_win(
    delta: QualityDeltaMetrics,
    *,
    speedup: float,
    latency_ratio: float,
) -> bool:
    quality_ok = (
        delta.delta_recall_at_1 >= -0.01
        and delta.delta_mrr_at_10 >= -0.01
        and delta.delta_ndcg_at_10 >= -0.01
    )
    runtime_advantage = speedup >= _STRONG_SPEEDUP or latency_ratio <= _MATERIAL_LATENCY_RATIO
    return quality_ok and runtime_advantage


def _is_bge_clear_win(
    delta: QualityDeltaMetrics,
    *,
    speedup: float,
    latency_ratio: float,
) -> bool:
    quality_lead = (
        delta.delta_recall_at_1 <= -_MATERIAL_QUALITY_DROP
        or delta.delta_mrr_at_10 <= -_MATERIAL_QUALITY_DROP
        or delta.delta_ndcg_at_10 <= -_MATERIAL_QUALITY_DROP
    )
    no_runtime_advantage = speedup < _MATERIAL_SPEEDUP and latency_ratio > _MATERIAL_LATENCY_RATIO
    if quality_lead and no_runtime_advantage:
        return True
    if (
        delta.delta_recall_at_1 < 0.0
        or delta.delta_mrr_at_10 < 0.0
        or delta.delta_ndcg_at_10 < 0.0
    ) and no_runtime_advantage:
        return True
    return False


def map_finalist_gate_to_decision(
    gate: FinalistQualificationGate,
) -> EmbeddingArenaDecision:
    if gate is FinalistQualificationGate.QWEN_CLEAR_WIN:
        return EmbeddingArenaDecision.PROMOTE_QWEN3_0_6B_CANDIDATE
    if gate is FinalistQualificationGate.BGE_CLEAR_WIN:
        return EmbeddingArenaDecision.KEEP_BGE_M3
    if gate is FinalistQualificationGate.QUALITY_REGRESSION:
        return EmbeddingArenaDecision.KEEP_BGE_M3
    if gate is FinalistQualificationGate.RUNTIME_REJECTED:
        return EmbeddingArenaDecision.NO_CLEAR_WINNER
    if gate is FinalistQualificationGate.AMBIGUOUS:
        return EmbeddingArenaDecision.MORE_EVIDENCE_REQUIRED
    return EmbeddingArenaDecision.NO_CLEAR_WINNER
