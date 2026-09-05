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


def _rank_key(item: StageBCandidateEvidence) -> tuple[float, float, str]:
    metrics = item.quality_metrics
    throughput = item.throughput_records_per_second
    if metrics is None or throughput is None:
        msg = "eligible Stage B evidence must include quality and throughput"
        raise ValueError(msg)
    return (-metrics.recall_at_10, -throughput, item.candidate_id)


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

    if len(eligible) <= max_finalists:
        return tuple(sorted(item.candidate_id for item in eligible))

    ranked = sorted(eligible, key=_rank_key)
    baseline_present = any(
        item.candidate_id == baseline_candidate_id for item in eligible
    )
    boost_threshold = baseline_throughput * THROUGHPUT_SPEEDUP_THRESHOLD

    finalist_ids: list[str] = []
    seen: set[str] = set()

    def _try_add(candidate_id: str) -> None:
        if candidate_id in seen:
            return
        if len(finalist_ids) >= max_finalists:
            return
        seen.add(candidate_id)
        finalist_ids.append(candidate_id)

    if baseline_present:
        _try_add(baseline_candidate_id)

    for item in ranked:
        if item.candidate_id == baseline_candidate_id:
            continue
        _try_add(item.candidate_id)

    for item in ranked:
        if item.candidate_id == baseline_candidate_id:
            continue
        if item.candidate_id in seen:
            continue
        throughput = item.throughput_records_per_second
        if throughput is not None and throughput >= boost_threshold:
            _try_add(item.candidate_id)

    if len(finalist_ids) > max_finalists:
        msg = "Stage C finalist selection exceeded max_finalists"
        raise RuntimeError(msg)

    return tuple(sorted(finalist_ids))
