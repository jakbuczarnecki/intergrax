# © Artur Czarnecki. All rights reserved.

"""Deterministic enterprise recall ranking (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.memory.contracts.enterprise_memory_record import MemoryTrustClass
from intergrax.memory.memory_temporal import is_memory_entry_active
from intergrax.memory.strategies.recall_models import (
    MemoryRankedCandidate,
    MemoryRankingRequest,
    MemoryRankingResult,
    MemoryRankingScore,
    MemoryRecallReasonCode,
)
from intergrax.memory.strategies.recall_validation import validate_ranking_result


def _parse_ts(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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


def _freshness_component(updated_at: str | None, created_at: str) -> float:
    anchor = _parse_ts(updated_at) or _parse_ts(created_at)
    if anchor is None:
        return 0.5
    now = datetime.now(tz=timezone.utc)
    age_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
    if age_days <= 1.0:
        return 1.0
    if age_days <= 30.0:
        return 0.75
    if age_days <= 180.0:
        return 0.55
    return 0.35


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
        scored: list[MemoryRankedCandidate] = []
        for candidate in request.candidates:
            record = candidate.record
            if record.lineage.superseded_by_memory_id and self._config.exclude_superseded:
                continue
            relevance = candidate.retrieval_score if candidate.retrieval_score is not None else 0.4
            relevance = max(0.0, min(1.0, relevance))
            trust_val = _trust_component(record.trust.trust_class, record.trust.confidence)
            temporal_val = 1.0 if is_memory_entry_active(record) else 0.0
            freshness_val = _freshness_component(record.updated_at, record.created_at)
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

        def sort_key(item: MemoryRankedCandidate) -> tuple[float, str, str]:
            record = item.candidate.record
            updated = record.updated_at or record.created_at or ""
            return (-item.score.total, updated, record.entry_id)

        scored.sort(key=sort_key)
        if request.top_k > 0:
            scored = scored[: request.top_k]
        result = MemoryRankingResult(ranked=tuple(scored))
        validate_ranking_result(request, result)
        return result
