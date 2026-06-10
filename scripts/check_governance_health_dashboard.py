#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-5.3 — governance health dashboard pane (GOV-PROD.1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    if not wiring.enabled or wiring.router is None or wiring.dashboard is None:
        print("governance dashboard requires unified observability wiring", file=sys.stderr)
        return 1
    paths = {route.path for route in wiring.router.routes}
    if "/ops/dashboard/governance" not in paths:
        print("governance dashboard must expose /ops/dashboard/governance", file=sys.stderr)
        return 1
    governance = wiring.dashboard.governance
    if not governance.tenant_isolation_verified:
        print("governance dashboard must report tenant isolation verified", file=sys.stderr)
        return 1
    print("OK: governance health dashboard on product hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
