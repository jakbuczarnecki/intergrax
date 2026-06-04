# © Artur Czarnecki. All rights reserved.

"""Tier-2 harness defaults — no Tier-3 imports (Phase DX-6.1)."""

from __future__ import annotations

from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)


def harness_production_mode() -> bool:
    """Lab/scaffold harness: relaxed governance (production_mode=False on RuntimeConfig)."""
    return False


__all__ = [
    "LabHarnessContext",
    "build_lab_agent_runtime_context",
    "default_reference_harness",
    "harness_production_mode",
]
