#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-28.4 — business agents K.1/K.2 certification + deploy readiness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "agents"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from problem_radar.problem_radar_agent import ProblemRadarAgent  # noqa: E402
from vendor_discovery.vendor_discovery_agent import VendorDiscoveryAgent  # noqa: E402
from intergrax.applications._shared.business_agent_deploy_wiring import (  # noqa: E402
    resolve_business_agent_deploy_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile  # noqa: E402

_BUSINESS_AGENTS = (ProblemRadarAgent, VendorDiscoveryAgent)


def main() -> int:
    wiring = resolve_business_agent_deploy_wiring(
        ApplicationEnvironmentProfile.product_defaults(),
        agent_factories=_BUSINESS_AGENTS,
    )
    if not wiring.enabled or wiring.report is None:
        print("business agent deploy wiring must be enabled on product hosts", file=sys.stderr)
        if wiring.report is not None:
            for item in wiring.report.results:
                for error in item.errors:
                    print(f"{item.agent_id}: {error}", file=sys.stderr)
        return 1
    if len(wiring.agent_ids) != 2:
        print("K.1/K.2 certification must cover two business agents", file=sys.stderr)
        return 1
    print(f"OK: business agent certification ({len(wiring.agent_ids)} agents deploy-ready)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
