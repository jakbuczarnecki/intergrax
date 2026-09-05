"""Deterministic Stage C finalist selection from complete Stage B evidence."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)

DEFAULT_MAX_STAGE_C_FINALISTS = 3
THROUGHPUT_SPEEDUP_THRESHOLD = 1.2


@dataclass(frozen=True, slots=True)
class StageBCandidateEvidence:
    candidate_id: str
    is_baseline: bool
    license_eligible: bool
    runtime_ok: bool
    throughput_records_per_second: float | None
    quality_metrics: RetrievalQualityMetrics | None


def select_stage_c_finalist_ids(
    stage_b_evidence: tuple[StageBCandidateEvidence, ...],
    *,
    baseline_candidate_id: str,
    baseline_throughput: float,
    max_finalists: int = DEFAULT_MAX_STAGE_C_FINALISTS,
) -> tuple[str, ...]:
    """Select Stage C finalists deterministically from the complete Stage B set."""
    if max_finalists <= 0:
        msg = "max_finalists must be > 0"
        raise ValueError(msg)

    eligible = [
        item
        for item in stage_b_evidence
        if item.runtime_ok
        and item.license_eligible
        and item.quality_metrics is not None
        and item.throughput_records_per_second is not None
        and item.throughput_records_per_second > 0.0
    ]

    def _rank_key(item: StageBCandidateEvidence) -> tuple[float, float, str]:
        metrics = item.quality_metrics
        throughput = item.throughput_records_per_second
        if metrics is None or throughput is None:
            msg = "eligible Stage B evidence must include quality and throughput"
            raise ValueError(msg)
        return (-metrics.recall_at_10, -throughput, item.candidate_id)

    ranked = sorted(eligible, key=_rank_key)

    finalist_ids: list[str] = []
    baseline_present = any(
        item.candidate_id == baseline_candidate_id for item in eligible
    )
    if baseline_present:
        finalist_ids.append(baseline_candidate_id)

    for item in ranked:
        if item.candidate_id == baseline_candidate_id:
            continue
        if item.candidate_id in finalist_ids:
            continue
        if len(finalist_ids) >= max_finalists:
            break
        finalist_ids.append(item.candidate_id)

    for item in eligible:
        if item.candidate_id == baseline_candidate_id:
            continue
        if item.candidate_id in finalist_ids:
            continue
        throughput = item.throughput_records_per_second
        if throughput is not None and throughput >= baseline_throughput * THROUGHPUT_SPEEDUP_THRESHOLD:
            finalist_ids.append(item.candidate_id)

    if len(eligible) <= max_finalists:
        for item in eligible:
            if item.candidate_id not in finalist_ids:
                finalist_ids.append(item.candidate_id)

    return tuple(sorted(finalist_ids))
