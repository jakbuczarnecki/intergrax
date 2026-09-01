"""Multi-channel candidate handoff helpers."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ChannelCandidateBatch,
    MultiChannelCandidateCollection,
)


def collect_channel_candidates(
    *batches: ChannelCandidateBatch,
) -> MultiChannelCandidateCollection:
    """Merge independent channel batches without fusion or score normalization."""
    return MultiChannelCandidateCollection.from_channel_batches(*batches)
