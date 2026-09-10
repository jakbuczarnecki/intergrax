"""Pluggable offer-level fusion strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    MultiChannelCandidateCollection,
    ProductCandidate,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidate,
    FusedOfferCandidateCollection,
    OfferChannelEvidence,
    OfferFusionConfiguration,
    reciprocal_rank_contribution,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.errors import (
    OfferCandidateFusionError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_sort_key,
)

_CHANNEL_EVIDENCE_ORDER: tuple[RetrievalChannel, ...] = (
    RetrievalChannel.EXACT,
    RetrievalChannel.LEXICAL,
    RetrievalChannel.STRUCTURED,
    RetrievalChannel.VECTOR,
)
_CHANNEL_ORDER_INDEX = {channel: index for index, channel in enumerate(_CHANNEL_EVIDENCE_ORDER)}


class OfferCandidateFusionStrategy(Protocol):
    """Strategy seam for offer-level candidate fusion."""

    def fuse(
        self,
        candidates: MultiChannelCandidateCollection,
        *,
        limit: int,
    ) -> FusedOfferCandidateCollection:
        ...


@dataclass(frozen=True, slots=True)
class _AggregatedOffer:
    source_ref: SourceRecordRef
    evidence_by_channel: dict[RetrievalChannel, OfferChannelEvidence]

    @property
    def fusion_score(self) -> float:
        return sum(
            evidence.reciprocal_rank_contribution for evidence in self.evidence_by_channel.values()
        )

    @property
    def supporting_channel_count(self) -> int:
        return len(self.evidence_by_channel)

    @property
    def best_channel_rank(self) -> int:
        return min(evidence.channel_rank for evidence in self.evidence_by_channel.values())

    def ordered_evidence(self) -> tuple[OfferChannelEvidence, ...]:
        return tuple(
            self.evidence_by_channel[channel]
            for channel in _CHANNEL_EVIDENCE_ORDER
            if channel in self.evidence_by_channel
        )

    def sort_key(self) -> tuple[float, int, int, tuple[str, str, str]]:
        return (
            -self.fusion_score,
            -self.supporting_channel_count,
            self.best_channel_rank,
            source_ref_sort_key(self.source_ref),
        )


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusionStrategy:
    """Canonical rank-based fusion — raw channel scores never enter the formula."""

    configuration: OfferFusionConfiguration = OfferFusionConfiguration()

    def fuse(
        self,
        candidates: MultiChannelCandidateCollection,
        *,
        limit: int,
    ) -> FusedOfferCandidateCollection:
        if type(limit) is not int or limit <= 0:
            raise OfferCandidateFusionError("limit must be a positive int")
        if not candidates.candidates:
            return FusedOfferCandidateCollection(candidates=())

        aggregated = _aggregate_candidates(
            candidates.candidates,
            rrf_k=self.configuration.rrf_k,
        )
        ranked = sorted(aggregated.values(), key=lambda offer: offer.sort_key())
        fused_candidates: list[FusedOfferCandidate] = []
        for fused_rank, aggregated_offer in enumerate(ranked[:limit]):
            fused_candidates.append(
                FusedOfferCandidate(
                    source_ref=aggregated_offer.source_ref,
                    offer_id=aggregated_offer.source_ref.offer_id,
                    fused_rank=fused_rank,
                    fusion_score=aggregated_offer.fusion_score,
                    supporting_channel_count=aggregated_offer.supporting_channel_count,
                    evidence=aggregated_offer.ordered_evidence(),
                )
            )
        return FusedOfferCandidateCollection(candidates=tuple(fused_candidates))


def _aggregate_candidates(
    candidates: tuple[ProductCandidate, ...],
    *,
    rrf_k: int,
) -> dict[SourceRecordRef, _AggregatedOffer]:
    aggregated: dict[SourceRecordRef, _AggregatedOffer] = {}
    seen_channel_identity: set[tuple[SourceRecordRef, RetrievalChannel]] = set()

    for candidate in candidates:
        identity = (candidate.source_ref, candidate.channel)
        if identity in seen_channel_identity:
            raise OfferCandidateFusionError(
                "duplicate source offer within the same retrieval channel is forbidden"
            )
        seen_channel_identity.add(identity)

        contribution = reciprocal_rank_contribution(rank=candidate.rank, rrf_k=rrf_k)
        evidence = OfferChannelEvidence(
            channel=candidate.channel,
            channel_rank=candidate.rank,
            channel_score=candidate.channel_score,
            reciprocal_rank_contribution=contribution,
        )

        existing = aggregated.get(candidate.source_ref)
        if existing is None:
            aggregated[candidate.source_ref] = _AggregatedOffer(
                source_ref=candidate.source_ref,
                evidence_by_channel={candidate.channel: evidence},
            )
            continue

        if candidate.channel in existing.evidence_by_channel:
            raise OfferCandidateFusionError(
                "duplicate source offer within the same retrieval channel is forbidden"
            )
        updated_evidence = dict(existing.evidence_by_channel)
        updated_evidence[candidate.channel] = evidence
        aggregated[candidate.source_ref] = _AggregatedOffer(
            source_ref=existing.source_ref,
            evidence_by_channel=updated_evidence,
        )

    return aggregated
