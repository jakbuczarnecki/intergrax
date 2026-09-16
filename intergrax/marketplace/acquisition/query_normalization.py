# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Merge neutral capability need with discovery query without conflating contracts."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery


def discovery_query_for_machine_acquisition(
    need: CapabilityNeed,
    discovery_query: CapabilityDiscoveryQuery,
) -> CapabilityDiscoveryQuery:
    """Narrow discovery kinds using need — never widen query constraints."""
    if not need.kinds:
        return discovery_query
    if not discovery_query.kinds:
        return discovery_query.model_copy(update={"kinds": need.kinds})
    need_set = frozenset(need.kinds)
    query_set = frozenset(discovery_query.kinds)
    intersection = tuple(kind for kind in discovery_query.kinds if kind in need_set)
    if not intersection:
        raise ValueError("capability need kinds do not intersect discovery_query.kinds")
    if query_set - need_set:
        return discovery_query.model_copy(update={"kinds": intersection})
    return discovery_query


def effective_query_text(
    need: CapabilityNeed,
    query_text: str | None,
) -> str | None:
    if query_text is not None:
        return query_text
    return need.intent_summary


__all__ = ["discovery_query_for_machine_acquisition", "effective_query_text"]
