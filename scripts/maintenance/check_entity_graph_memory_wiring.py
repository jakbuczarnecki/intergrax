#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-15.3 — governed entity/temporal memory wiring gate."""

from __future__ import annotations

import sys

from intergrax.applications._shared.entity_graph_wiring import (
    resolve_entity_temporal_memory_capability,
)
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryCapability


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    if not env.memory_profile.enable_entity_graph_memory:
        print("product_defaults must enable entity graph memory", file=sys.stderr)
        return 1
    capability = resolve_entity_temporal_memory_capability(env)
    if capability is None:
        print("entity temporal capability must resolve for product profile", file=sys.stderr)
        return 1
    wiring = resolve_memory_platform_wiring(env)
    if wiring.entity_temporal_memory_capability is None:
        print("memory platform wiring must include entity temporal capability", file=sys.stderr)
        return 1

    if not isinstance(capability, EntityTemporalMemoryCapability):
        print("unexpected entity temporal capability type", file=sys.stderr)
        return 1

    print("OK: governed entity temporal memory wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
