# © Artur Czarnecki. All rights reserved.

"""Tier-2 harness defaults — no Tier-3 imports (Phase DX-6.1)."""

from __future__ import annotations

from intergrax.agents.reference_harness import (
    LabHarnessContext,
    default_reference_harness,
)


def harness_production_mode() -> bool:
    """Lab/scaffold harness constant: relaxed governance (production_mode=False on RuntimeConfig).

    Tier-2 ``build_context`` implementations must not use it: they take the posture from the host
    (see FC-1 / NT-BND-05-1). It remains the documented LAB value for lab-only callers.
    """
    return False


__all__ = [
    "LabHarnessContext",
    "default_reference_harness",
    "harness_production_mode",
]
