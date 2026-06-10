#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-27.3 — agent simulator on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.agent_simulator_wiring import resolve_agent_simulator_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_agent_simulator_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable agent simulator routes", file=sys.stderr)
        return 1
    if wiring.router is None:
        print("agent simulator router missing", file=sys.stderr)
        return 1
    paths = {route.path for route in wiring.router.routes}
    if "/mvp/simulate" not in paths or "/mvp/replay" not in paths:
        print("agent simulator must expose /mvp/simulate and /mvp/replay", file=sys.stderr)
        return 1
    print("OK: agent simulator wiring on product hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
