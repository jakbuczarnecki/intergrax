"""Finalist qualification decision gate — baseline vs challenger on controlled sample."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaStageStatus,
    EmbeddingArenaVerdict,
    FinalistQualificationGate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    FinalistQualificationSelectionError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.finalist_qualification import (
    FinalistQualificationSelection,
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


def _resolve_candidate_result(
    results: tuple[CandidateArenaResult, ...],
    *,
    candidate_id: str,
    role: str,
) -> CandidateArenaResult:
    matches = tuple(item for item in results if item.candidate_id == candidate_id)
    if len(matches) != 1:
        msg = (
            f"{role} candidate {candidate_id!r} not found in qualification results "
            f"(expected exactly one match)"
        )
        raise FinalistQualificationSelectionError(msg)
    return matches[0]


def classify_finalist_qualification_gate(
    results: tuple[CandidateArenaResult, ...],
    selection: FinalistQualificationSelection,
) -> tuple[FinalistQualificationGate, str]:
    baseline = _resolve_candidate_result(
        results,
        candidate_id=selection.baseline_candidate_id,
        role="baseline",
    )
    challenger = _resolve_candidate_result(
        results,
        candidate_id=selection.challenger_candidate_id,
        role="challenger",
    )
    if _runtime_rejected(baseline) or _runtime_rejected(challenger):
        return (
            FinalistQualificationGate.RUNTIME_REJECTED,
            (
                f"one or more finalists failed runtime qualification "
                f"(baseline={selection.baseline_candidate_id}, "
                f"challenger={selection.challenger_candidate_id})"
            ),
        )
    if not _quality_metrics_ready(baseline) or not _quality_metrics_ready(challenger):
        return (
            FinalistQualificationGate.RUNTIME_REJECTED,
            (
                f"quality metrics unavailable for finalist comparison "
                f"(baseline={selection.baseline_candidate_id}, "
                f"challenger={selection.challenger_candidate_id})"
            ),
        )

    assert baseline.quality_metrics is not None
    assert challenger.quality_metrics is not None
    delta = compute_quality_delta(challenger.quality_metrics, baseline.quality_metrics)
    if not quality_non_regression_gate(delta):
        return (
            FinalistQualificationGate.QUALITY_REGRESSION,
            (
                f"challenger {selection.challenger_candidate_id} fails quality "
                f"non-regression vs baseline {selection.baseline_candidate_id} "
                f"on recall@10 or mrr@10"
            ),
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
            (
                f"decision-critical quality metrics and throughput remain within "
                f"sample-noise band (baseline={selection.baseline_candidate_id}, "
                f"challenger={selection.challenger_candidate_id})"
            ),
        )
    if _is_challenger_clear_win(delta, speedup=speedup, latency_ratio=latency_ratio):
        return (
            FinalistQualificationGate.CHALLENGER_CLEAR_WIN,
            (
                f"challenger {selection.challenger_candidate_id} matches or exceeds "
                f"baseline {selection.baseline_candidate_id} on recall@1/mrr/ndcg "
                f"with runtime advantage"
            ),
        )
    if _is_baseline_clear_win(delta, speedup=speedup, latency_ratio=latency_ratio):
        return (
            FinalistQualificationGate.BASELINE_CLEAR_WIN,
            (
                f"baseline {selection.baseline_candidate_id} leads decision-critical "
                f"quality without challenger {selection.challenger_candidate_id} "
                f"runtime advantage"
            ),
        )
    return (
        FinalistQualificationGate.MORE_EVIDENCE_REQUIRED,
        (
            f"no clear finalist winner on combined quality and runtime evidence "
            f"(baseline={selection.baseline_candidate_id}, "
            f"challenger={selection.challenger_candidate_id})"
        ),
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


def _is_challenger_clear_win(
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


def _is_baseline_clear_win(
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
