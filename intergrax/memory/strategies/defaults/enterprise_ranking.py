# © Artur Czarnecki. All rights reserved.

"""Deterministic enterprise recall ranking (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryTrustClass,
    parse_memory_record_timestamp,
)
from intergrax.memory.memory_temporal import is_memory_entry_active
from intergrax.memory.strategies.recall_models import (
    MemoryRankedCandidate,
    MemoryRankingRequest,
    MemoryRankingResult,
    MemoryRankingScore,
    MemoryRecallReasonCode,
)
from intergrax.memory.strategies.recall_validation import validate_ranking_result


def _optional_record_timestamp(field_name: str, value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return parse_memory_record_timestamp(field_name, text)
    except ValueError:
        return None


def _timestamps_comparable(left: datetime, right: datetime) -> bool:
    left_aware = left.tzinfo is not None
    right_aware = right.tzinfo is not None
    return left_aware == right_aware


def _parse_as_of(as_of_iso: str | None) -> datetime | None:
    if as_of_iso is None:
        return None
    text = as_of_iso.strip()
    if not text:
        return None
    return parse_memory_record_timestamp("as_of", text)


def _trust_component(trust_class: MemoryTrustClass, confidence: float | None) -> float:
    base = {
        MemoryTrustClass.USER_EXPLICIT: 0.85,
        MemoryTrustClass.EXTERNAL_SOURCE: 0.7,
        MemoryTrustClass.SYSTEM_GENERATED: 0.55,
        MemoryTrustClass.MODEL_INFERENCE: 0.45,
        MemoryTrustClass.UNKNOWN: 0.35,
    }.get(trust_class, 0.35)
    if confidence is not None:
        return 0.5 * base + 0.5 * max(0.0, min(1.0, confidence))
    return base


def _freshness_component(
    updated_at: str | None,
    created_at: str,
    as_of: datetime | None,
) -> float:
    if as_of is None:
        return 0.5
    anchor = _optional_record_timestamp("updated_at", updated_at) or _optional_record_timestamp(
        "created_at", created_at
    )
    if anchor is None:
        return 0.5
    if not _timestamps_comparable(anchor, as_of):
        return 0.5
    age_days = max(0.0, (as_of - anchor).total_seconds() / 86400.0)
    if age_days <= 1.0:
        return 1.0
    if age_days <= 30.0:
        return 0.75
    if age_days <= 180.0:
        return 0.55
    return 0.35


def _naive_chronological_ordinal(dt: datetime) -> float:
    seconds_since_midnight = (
        dt.hour * 3600.0
        + dt.minute * 60.0
        + dt.second
        + dt.microsecond / 1_000_000.0
    )
    return dt.toordinal() * 86400.0 + seconds_since_midnight


def _recency_ordinal(updated_at: str | None, created_at: str) -> float:
    """Sortable recency key: larger means newer; missing timestamps sort last."""
    anchor = _optional_record_timestamp("updated_at", updated_at) or _optional_record_timestamp(
        "created_at", created_at
    )
    if anchor is None:
        return float("-inf")
    if anchor.tzinfo is not None:
        return anchor.timestamp()
    return _naive_chronological_ordinal(anchor)


@dataclass(frozen=True, slots=True)
class EnterpriseMemoryRankingConfig:
    relevance_weight: float = 0.55
    trust_weight: float = 0.15
    temporal_weight: float = 0.15
    freshness_weight: float = 0.10
    superseded_penalty: float = 0.35
    exclude_superseded: bool = True


class EnterpriseMemoryRankingStrategy:
    strategy_id = "enterprise_deterministic_ranking_v1"

    def __init__(self, config: EnterpriseMemoryRankingConfig | None = None) -> None:
        self._config = config or EnterpriseMemoryRankingConfig()

    def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
        as_of = _parse_as_of(request.as_of_iso)
        scored: list[MemoryRankedCandidate] = []
        for candidate in request.candidates:
            record = candidate.record
            if record.lineage.superseded_by_memory_id and self._config.exclude_superseded:
                continue
            relevance = candidate.retrieval_score if candidate.retrieval_score is not None else 0.4
            relevance = max(0.0, min(1.0, relevance))
            trust_val = _trust_component(record.trust.trust_class, record.trust.confidence)
            temporal_val = (
                1.0
                if is_memory_entry_active(record, as_of=as_of)
                else 0.0
            )
            freshness_val = _freshness_component(record.updated_at, record.created_at, as_of)
            supersession_adj = 0.0
            if record.lineage.superseded_by_memory_id:
                supersession_adj = -self._config.superseded_penalty
            total = (
                self._config.relevance_weight * relevance
                + self._config.trust_weight * trust_val
                + self._config.temporal_weight * temporal_val
                + self._config.freshness_weight * freshness_val
                + supersession_adj
            )
            reason_codes: list[MemoryRecallReasonCode] = []
            if relevance >= 0.7:
                reason_codes.append(MemoryRecallReasonCode.HIGH_RELEVANCE)
            if record.trust.trust_class is MemoryTrustClass.USER_EXPLICIT:
                reason_codes.append(MemoryRecallReasonCode.TRUSTED_SOURCE)
            if freshness_val >= 0.75:
                reason_codes.append(MemoryRecallReasonCode.FRESHER_RECORD)
            if record.lineage.superseded_by_memory_id:
                reason_codes.append(MemoryRecallReasonCode.SUPERSEDED_RECORD)
            if temporal_val == 0.0:
                reason_codes.append(MemoryRecallReasonCode.TEMPORAL_INACTIVE)
            scored.append(
                MemoryRankedCandidate(
                    candidate=candidate,
                    score=MemoryRankingScore(
                        total=total,
                        relevance=relevance,
                        trust=trust_val,
                        temporal=temporal_val,
                        freshness=freshness_val,
                        supersession_adjustment=supersession_adj,
                    ),
                    reason_codes=tuple(reason_codes),
                )
            )

        def sort_key(item: MemoryRankedCandidate) -> tuple[float, float, str]:
            record = item.candidate.record
            return (
                -item.score.total,
                -_recency_ordinal(record.updated_at, record.created_at),
                record.entry_id,
            )

        scored.sort(key=sort_key)
        if request.top_k > 0:
            scored = scored[: request.top_k]
        result = MemoryRankingResult(ranked=tuple(scored))
        validate_ranking_result(request, result)
        return result
