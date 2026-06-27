#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 capability graph wiring and catalog baseline (Phase CG-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.capability_graph import build_catalog_capability_graph
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest


def main() -> int:
    catalog = build_catalog_capability_graph()
    if not catalog.nodes:
        print("catalog capability graph has no nodes")
        return 1
    if not catalog.edges:
        print("catalog capability graph has no edges")
        return 1

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cg.audit")
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    if wiring.capability_graph is None:
        print("wire_application_environment did not materialize capability_graph")
        return 1
    if not wiring.capability_graph.graph.nodes:
        print("environment capability graph is empty")
        return 1

    print("harness capability graph wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
