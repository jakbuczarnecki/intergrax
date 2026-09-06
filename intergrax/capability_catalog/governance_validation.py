# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed governed output validation (CAPABILITY-CATALOG-1 Stage 5)."""

from __future__ import annotations

from intergrax.capability_catalog.errors import CapabilityGovernanceError
from intergrax.capability_catalog.governed_result import GovernedDiscoveryResult
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition

_NON_ALLOWED_AVAILABILITY = frozenset(
    {
        AvailabilityDisposition.BLOCKED,
        AvailabilityDisposition.UNAVAILABLE,
        AvailabilityDisposition.SCOPE_UNAVAILABLE,
    },
)


def validate_governed_output(
    *,
    input_ranked: tuple[RankedCapabilityCandidate, ...],
    result: GovernedDiscoveryResult,
) -> None:
    """Reject governed output that mutates, drops, duplicates, or elevates candidates."""
    expected_count = len(input_ranked)
    actual_count = len(result.allowed) + len(result.blocked)
    if actual_count != expected_count:
        raise CapabilityGovernanceError(
            "governed output must contain exactly the same number of candidates "
            f"as input (expected {expected_count}, got {actual_count})",
        )

    input_by_key = {
        candidate.identity.sort_key: candidate for candidate in input_ranked
    }
    seen_keys: set[tuple[str, str, str, str]] = set()
    allowed_keys: set[tuple[str, str, str, str]] = set()
    blocked_keys: set[tuple[str, str, str, str]] = set()

    for partition_label, items in (
        ("allowed", result.allowed),
        ("blocked", result.blocked),
    ):
        for item in items:
            key = item.ranked.identity.sort_key
            if key not in input_by_key:
                raise CapabilityGovernanceError(
                    f"governed {partition_label} contains unknown candidate identity",
                )
            if key in seen_keys:
                raise CapabilityGovernanceError(
                    f"governed output contains duplicate candidate identity in {partition_label}",
                )
            seen_keys.add(key)
            if partition_label == "allowed":
                allowed_keys.add(key)
            else:
                blocked_keys.add(key)

            original = input_by_key[key]
            if item.ranked != original:
                raise CapabilityGovernanceError(
                    "governed output must not mutate ranked candidate identity, "
                    "provenance, availability, or ranking evidence",
                )

            if partition_label == "allowed":
                if item.availability in _NON_ALLOWED_AVAILABILITY:
                    raise CapabilityGovernanceError(
                        "governed output must not elevate blocked/unavailable candidates",
                    )

    if allowed_keys & blocked_keys:
        raise CapabilityGovernanceError(
            "governed output allowed and blocked partitions must be disjoint",
        )

    if seen_keys != set(input_by_key):
        raise CapabilityGovernanceError(
            "governed output is missing one or more input candidates",
        )

    _validate_rank_order_preserved(
        input_ranked=input_ranked,
        result=result,
    )


def _validate_rank_order_preserved(
    *,
    input_ranked: tuple[RankedCapabilityCandidate, ...],
    result: GovernedDiscoveryResult,
) -> None:
    input_order = [candidate.identity.sort_key for candidate in input_ranked]

    def _relative_order(
        items: tuple[RankedCapabilityCandidate, ...],
    ) -> list[tuple[str, str, str, str]]:
        return [item.identity.sort_key for item in items]

    allowed_order = _relative_order(tuple(item.ranked for item in result.allowed))
    blocked_order = _relative_order(tuple(item.ranked for item in result.blocked))

    def _is_subsequence(
        subsequence: list[tuple[str, str, str, str]],
        sequence: list[tuple[str, str, str, str]],
    ) -> bool:
        if not subsequence:
            return True
        index = 0
        for key in sequence:
            if key == subsequence[index]:
                index += 1
                if index == len(subsequence):
                    return True
        return False

    if not _is_subsequence(allowed_order, input_order):
        raise CapabilityGovernanceError(
            "governed allowed partition must preserve Stage-4 rank order",
        )
    if not _is_subsequence(blocked_order, input_order):
        raise CapabilityGovernanceError(
            "governed blocked partition must preserve Stage-4 rank order",
        )
