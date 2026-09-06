# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed ranked output validation (CAPABILITY-CATALOG-1 Stage 4)."""

from __future__ import annotations

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.errors import CapabilityRankingError
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.contracts.capability_catalog._validation import require_non_empty_text


def validate_ranked_output(
    *,
    input_candidates: tuple[CapabilityDiscoveryCandidate, ...],
    ranked: tuple[RankedCapabilityCandidate, ...],
    ranker_id: str,
) -> None:
    """Reject ranker output that mutates, drops, duplicates, or re-identifies candidates."""
    require_non_empty_text(ranker_id, label="ranker_id")
    expected_count = len(input_candidates)
    if len(ranked) != expected_count:
        raise CapabilityRankingError(
            "ranker output must contain exactly the same number of candidates "
            f"as input (expected {expected_count}, got {len(ranked)})",
        )

    input_by_key = {
        candidate.identity.sort_key: candidate for candidate in input_candidates
    }
    seen_keys: set[tuple[str, str, str, str]] = set()
    seen_positions: set[int] = set()

    for item in ranked:
        if item.evidence.ranker_id != ranker_id:
            raise CapabilityRankingError(
                "ranker output evidence.ranker_id must match the active ranker",
            )
        key = item.candidate.identity.sort_key
        if key not in input_by_key:
            raise CapabilityRankingError(
                "ranker output contains unknown candidate identity",
            )
        if key in seen_keys:
            raise CapabilityRankingError(
                "ranker output contains duplicate candidate identity",
            )
        seen_keys.add(key)

        original = input_by_key[key]
        if item.candidate != original:
            raise CapabilityRankingError(
                "ranker output must not mutate candidate identity, provenance, or availability",
            )

        position = item.evidence.rank_position
        if position in seen_positions:
            raise CapabilityRankingError(
                "ranker output contains duplicate rank_position values",
            )
        seen_positions.add(position)

    if seen_positions != set(range(1, expected_count + 1)):
        raise CapabilityRankingError(
            "ranker output rank_position values must be contiguous 1..N",
        )

    if seen_keys != set(input_by_key):
        raise CapabilityRankingError(
            "ranker output is missing one or more input candidates",
        )
