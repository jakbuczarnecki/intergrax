# © Artur Czarnecki. All rights reserved.

"""Orchestration capability tokens — routing labels, not agent registry capabilities (ORCH-CONFIG)."""

from __future__ import annotations

from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec


def orchestration_capabilities_from_graph_spec(
    spec: ApplicationGraphSpec | None,
) -> frozenset[str]:
    """Explicit trigger capabilities declared on a Tier-3 graph spec."""
    if spec is None:
        return frozenset()
    return frozenset(item.strip() for item in spec.trigger_capabilities if item.strip())


def is_orchestration_capability(
    capability: str,
    *,
    trigger_capabilities: frozenset[str],
    pipeline_capability_suffix: str = ".pipeline",
) -> bool:
    """
    True when ``capability`` is a harness orchestration token (graph seed / rules route).

    Orchestration tokens are **not** required on agent contracts — ``GraphSpecSeedingPlanner``
    binds explicit ``agent_id`` values from ``ApplicationGraphSpec``.
    """
    normalized = (capability or "").strip()
    if not normalized:
        return False
    if normalized in trigger_capabilities:
        return True
    suffix = (pipeline_capability_suffix or "").strip()
    if suffix and normalized.endswith(suffix):
        return True
    return False
