"""Arena verdict and decision logic — dominance gates, not weighted scores."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    EmbeddingArenaVerdict,
    SpeedupBand,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    CandidateArenaResult,
    QualityDeltaMetrics,
    RetrievalQualityMetrics,
    SpeedupEstimate,
)


def classify_speedup_band(speedup: float) -> SpeedupBand:
    if speedup < 1.5:
        return SpeedupBand.UNDER_1_5X
    if speedup < 2.0:
        return SpeedupBand.BAND_1_5_TO_2X
    if speedup <= 4.0:
        return SpeedupBand.BAND_2_TO_4X
    return SpeedupBand.OVER_4X


def compute_speedup_estimate(
    *,
    candidate_records_per_second: float,
    baseline_records_per_second: float,
    baseline_embedding_hours: float | None,
) -> SpeedupEstimate:
    if candidate_records_per_second <= 0.0 or baseline_records_per_second <= 0.0:
        msg = "records_per_second values must be > 0"
        raise ValueError(msg)
    speedup = candidate_records_per_second / baseline_records_per_second
    hours_saved = None
    if baseline_embedding_hours is not None and baseline_embedding_hours > 0.0:
        candidate_hours = baseline_embedding_hours / speedup
        hours_saved = baseline_embedding_hours - candidate_hours
    return SpeedupEstimate(
        speedup_vs_baseline=speedup,
        speedup_band=classify_speedup_band(speedup),
        hours_saved_vs_baseline=hours_saved,
    )


def compute_quality_delta(
    candidate: RetrievalQualityMetrics,
    baseline: RetrievalQualityMetrics,
) -> QualityDeltaMetrics:
    return QualityDeltaMetrics(
        delta_recall_at_1=candidate.recall_at_1 - baseline.recall_at_1,
        delta_recall_at_5=candidate.recall_at_5 - baseline.recall_at_5,
        delta_recall_at_10=candidate.recall_at_10 - baseline.recall_at_10,
        delta_mrr_at_10=candidate.mrr_at_10 - baseline.mrr_at_10,
        delta_ndcg_at_10=candidate.ndcg_at_10 - baseline.ndcg_at_10,
    )


def quality_non_regression_gate(
    delta: QualityDeltaMetrics,
    *,
    max_recall_at_10_drop: float = 0.05,
    max_mrr_at_10_drop: float = 0.05,
) -> bool:
    if delta.delta_recall_at_10 < -max_recall_at_10_drop:
        return False
    if delta.delta_mrr_at_10 < -max_mrr_at_10_drop:
        return False
    return True


def classify_candidate_verdict(
    *,
    is_baseline: bool,
    license_eligible: bool,
    runtime_ok: bool,
    correctness_ok: bool,
    quality_delta: QualityDeltaMetrics | None,
    speedup: SpeedupEstimate | None,
    long_input_regression: bool,
) -> EmbeddingArenaVerdict:
    if is_baseline:
        return EmbeddingArenaVerdict.BASELINE
    if not license_eligible:
        return EmbeddingArenaVerdict.REJECTED_LICENSE
    if not runtime_ok:
        return EmbeddingArenaVerdict.REJECTED_RUNTIME
    if not correctness_ok:
        return EmbeddingArenaVerdict.REJECTED_CORRECTNESS
    if quality_delta is not None and not quality_non_regression_gate(quality_delta):
        return EmbeddingArenaVerdict.REJECTED_QUALITY
    if long_input_regression:
        return EmbeddingArenaVerdict.QUALIFIED_TRADEOFF
    if speedup is not None and speedup.speedup_vs_baseline >= 2.0:
        if quality_delta is not None and quality_non_regression_gate(quality_delta):
            return EmbeddingArenaVerdict.WINNER_CANDIDATE
    if speedup is not None and speedup.speedup_vs_baseline >= 1.5:
        return EmbeddingArenaVerdict.QUALIFIED_TRADEOFF
    return EmbeddingArenaVerdict.QUALIFIED


def decide_arena_outcome(
    results: tuple[CandidateArenaResult, ...],
) -> tuple[EmbeddingArenaDecision, str, tuple[str, ...]]:
    baseline = next((item for item in results if item.verdict is EmbeddingArenaVerdict.BASELINE), None)
    winners = [
        item
        for item in results
        if item.verdict is EmbeddingArenaVerdict.WINNER_CANDIDATE
    ]
    tradeoffs = [
        item
        for item in results
        if item.verdict is EmbeddingArenaVerdict.QUALIFIED_TRADEOFF
    ]

    if winners:
        winner = winners[0]
        decision_map = {
            "qwen3-0.6b": EmbeddingArenaDecision.PROMOTE_QWEN3_0_6B_CANDIDATE,
            "nomic-v2-moe": EmbeddingArenaDecision.PROMOTE_NOMIC_V2_CANDIDATE,
            "e5-large-instruct": EmbeddingArenaDecision.PROMOTE_E5_INSTRUCT_CANDIDATE,
        }
        decision = decision_map.get(winner.candidate_id, EmbeddingArenaDecision.MORE_EVIDENCE_REQUIRED)
        rationale = (
            f"{winner.candidate_id} meets quality non-regression with material speedup "
            f"({winner.speedup_estimate.speedup_vs_baseline:.2f}x)"
            if winner.speedup_estimate is not None
            else f"{winner.candidate_id} meets winner gates"
        )
        finalists = tuple(item.candidate_id for item in winners[:2])
        return decision, rationale, finalists

    if tradeoffs:
        best = max(
            tradeoffs,
            key=lambda item: item.speedup_estimate.speedup_vs_baseline
            if item.speedup_estimate is not None
            else 0.0,
        )
        rationale = (
            f"No clear winner; best tradeoff candidate {best.candidate_id} "
            "requires operator review before promotion"
        )
        finalists = tuple(item.candidate_id for item in tradeoffs[:2])
        return EmbeddingArenaDecision.MORE_EVIDENCE_REQUIRED, rationale, finalists

    if baseline is not None:
        return (
            EmbeddingArenaDecision.KEEP_BGE_M3,
            "No challenger materially improved throughput without quality regression",
            ("bge-m3",),
        )

    return (
        EmbeddingArenaDecision.NO_CLEAR_WINNER,
        "Insufficient successful candidate evidence",
        (),
    )
