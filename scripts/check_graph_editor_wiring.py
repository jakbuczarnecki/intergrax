#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-27.4 — visual builder / graph editor routes."""

from __future__ import annotations

import sys

from intergrax.applications._shared.graph_editor_wiring import resolve_graph_editor_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_graph_editor_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable graph editor routes", file=sys.stderr)
        return 1
    if wiring.router is None:
        print("graph editor router missing", file=sys.stderr)
        return 1
    paths = {route.path for route in wiring.router.routes}
    if "/ops/graph/render" not in paths or "/ops/graph/template" not in paths:
        print("graph editor must expose /ops/graph/render and /ops/graph/template", file=sys.stderr)
        return 1
    print("OK: graph editor wiring on product hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
