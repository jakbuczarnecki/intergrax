# © Artur Czarnecki. All rights reserved.

"""Orchestration capability tokens — routing labels, not agent registry capabilities."""

from __future__ import annotations

from collections.abc import Sequence


def orchestration_capabilities_from_triggers(
    trigger_capabilities: Sequence[str] | None,
) -> frozenset[str]:
    """Explicit trigger capabilities declared on a Tier-3 graph spec."""
    if not trigger_capabilities:
        return frozenset()
    return frozenset(item.strip() for item in trigger_capabilities if item.strip())


def is_orchestration_capability(
    capability: str,
    *,
    trigger_capabilities: frozenset[str],
    pipeline_capability_suffix: str = ".pipeline",
) -> bool:
    """True when ``capability`` is a harness orchestration token."""
    normalized = (capability or "").strip()
    if not normalized:
        return False
    if normalized in trigger_capabilities:
        return True
    suffix = (pipeline_capability_suffix or "").strip()
    if suffix and normalized.endswith(suffix):
        return True
    return False
