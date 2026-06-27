#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-21.3 — unified product observability dashboard (GOV-PROD.1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "agents", ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.applications._shared.product_observability_dashboard_wiring import (  # noqa: E402
    resolve_product_observability_dashboard_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile  # noqa: E402


def main() -> int:
    wiring = resolve_product_observability_dashboard_wiring(
        ApplicationEnvironmentProfile.product_defaults(),
        repo_root=ROOT,
    )
    if not wiring.enabled:
        print("product host must enable unified observability dashboard", file=sys.stderr)
        return 1
    if wiring.router is None or wiring.dashboard is None:
        print("unified observability dashboard router missing", file=sys.stderr)
        return 1
    paths = {route.path for route in wiring.router.routes}
    if "/ops/dashboard/health" not in paths:
        print("dashboard must expose /ops/dashboard/health", file=sys.stderr)
        return 1
    if "/ops/dashboard/unified" not in paths:
        print("dashboard must expose /ops/dashboard/unified", file=sys.stderr)
        return 1
    print(f"OK: unified product observability dashboard ({wiring.dashboard.host_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
