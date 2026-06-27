#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-27.2 — replay environment HTTP API on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.replay_environment_wiring import (
    resolve_replay_environment_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_replay_environment_wiring(
        ApplicationEnvironmentProfile.product_defaults()
    )
    if not wiring.enabled:
        print("product host must enable replay environment routes", file=sys.stderr)
        return 1
    if wiring.router is None:
        print("replay environment router missing", file=sys.stderr)
        return 1
    paths = {route.path for route in wiring.router.routes}
    if "/harness/replay" not in paths:
        print("replay environment must expose /harness/replay", file=sys.stderr)
        return 1
    print("OK: replay environment wiring on product hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
