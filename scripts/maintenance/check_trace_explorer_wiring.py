#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-27.1 — Trace Explorer on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.trace_explorer_wiring import resolve_trace_explorer_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_trace_explorer_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable trace explorer routes", file=sys.stderr)
        return 1
    if wiring.router is None:
        print("trace explorer router missing", file=sys.stderr)
        return 1
    paths = {route.path for route in wiring.router.routes}
    if "/ops/trace/health" not in paths:
        print("trace explorer must expose /ops/trace/health", file=sys.stderr)
        return 1
    print("OK: trace explorer wiring on product hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
