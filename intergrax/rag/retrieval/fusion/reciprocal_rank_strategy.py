# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Default platform RRF strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.rag.retrieval.fusion.contracts import (
    RECIPROCAL_RANK_FUSION_STRATEGY_ID,
    FusedRankedCandidate,
    RankFusionChannelEvidence,
    RankFusionConfiguration,
    RankFusionResult,
    RankedRetrievalCandidate,
    RankedRetrievalChannel,
)
from intergrax.rag.retrieval.fusion.errors import RankFusionContractError
from intergrax.rag.retrieval.fusion.reciprocal_rank import reciprocal_rank_contribution

TPayload = TypeVar("TPayload")


@dataclass(frozen=True, slots=True)
class _AggregatedCandidate(Generic[TPayload]):
    candidate_id: str
    payload: TPayload | None
    evidence_by_channel: dict[str, RankFusionChannelEvidence]

    @property
    def fusion_score(self) -> float:
        return sum(item.reciprocal_rank_contribution for item in self.evidence_by_channel.values())

    @property
    def supporting_channel_count(self) -> int:
        return len(self.evidence_by_channel)

    @property
    def best_channel_rank(self) -> int:
        return min(item.channel_rank for item in self.evidence_by_channel.values())

    def ordered_evidence(self) -> tuple[RankFusionChannelEvidence, ...]:
        return tuple(
            self.evidence_by_channel[key]
            for key in sorted(self.evidence_by_channel)
        )

    def sort_key(self) -> tuple[float, int, str]:
        return (
            -self.fusion_score,
            self.best_channel_rank,
            self.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusionStrategy(Generic[TPayload]):
    """Canonical rank-based fusion — source scores never enter the formula."""

    configuration: RankFusionConfiguration = RankFusionConfiguration()

    @property
    def strategy_id(self) -> str:
        return RECIPROCAL_RANK_FUSION_STRATEGY_ID

    def fuse(
        self,
        channels: tuple[RankedRetrievalChannel[TPayload], ...],
        *,
        limit: int | None = None,
    ) -> RankFusionResult[TPayload]:
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise RankFusionContractError("limit must be a positive int when provided")

        if not channels:
            return RankFusionResult(candidates=())

        aggregated = _aggregate_channels(channels, rrf_k=self.configuration.rrf_k)
        ranked = sorted(aggregated.values(), key=lambda item: item.sort_key())
        selected = ranked[:limit] if limit is not None else ranked

        fused: list[FusedRankedCandidate[TPayload]] = []
        for fused_rank, item in enumerate(selected):
            fused.append(
                FusedRankedCandidate(
                    candidate_id=item.candidate_id,
                    fused_rank=fused_rank,
                    fusion_score=item.fusion_score,
                    supporting_channel_count=item.supporting_channel_count,
                    evidence=item.ordered_evidence(),
                    payload=item.payload,
                )
            )
        return RankFusionResult(candidates=tuple(fused))


def _aggregate_channels(
    channels: tuple[RankedRetrievalChannel[TPayload], ...],
    *,
    rrf_k: int,
) -> dict[str, _AggregatedCandidate[TPayload]]:
    aggregated: dict[str, _AggregatedCandidate[TPayload]] = {}
    seen_in_channel: set[tuple[str, str]] = set()

    for channel in channels:
        for candidate in channel.candidates:
            identity = (channel.channel_key, candidate.candidate_id)
            if identity in seen_in_channel:
                raise RankFusionContractError(
                    "duplicate candidate_id within the same retrieval channel is forbidden"
                )
            seen_in_channel.add(identity)

            contribution = reciprocal_rank_contribution(
                rank=candidate.rank,
                rrf_k=rrf_k,
            )
            evidence = RankFusionChannelEvidence(
                channel_key=channel.channel_key,
                channel_rank=candidate.rank,
                reciprocal_rank_contribution=contribution,
            )

            existing = aggregated.get(candidate.candidate_id)
            if existing is None:
                aggregated[candidate.candidate_id] = _AggregatedCandidate(
                    candidate_id=candidate.candidate_id,
                    payload=candidate.payload,
                    evidence_by_channel={channel.channel_key: evidence},
                )
                continue

            if channel.channel_key in existing.evidence_by_channel:
                raise RankFusionContractError(
                    "duplicate candidate_id within the same retrieval channel is forbidden"
                )

            updated = dict(existing.evidence_by_channel)
            updated[channel.channel_key] = evidence
            aggregated[candidate.candidate_id] = _AggregatedCandidate(
                candidate_id=existing.candidate_id,
                payload=existing.payload,
                evidence_by_channel=updated,
            )

    return aggregated
