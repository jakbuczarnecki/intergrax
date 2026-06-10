#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-28.3 — LKW hybrid daemon scaffold (CFG-14)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from intergrax.applications._shared.lkw_hybrid_daemon_wiring import resolve_lkw_hybrid_daemon_wiring  # noqa: E402
from intergrax.applications.contracts.environment_profile import (  # noqa: E402
    ApplicationEnvironmentProfile,
    HostDeploymentProfile,
)


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults(
        profile_id="local_workspace.product",
        skill_bundles=["harness"],
    ).model_copy(
        update={
            "host_deployment_profile": HostDeploymentProfile(
                lkw_hybrid_daemon_enabled=True,
                lkw_daemon_bind_host="127.0.0.1",
                lkw_daemon_port=8020,
            ),
        }
    )
    wiring = resolve_lkw_hybrid_daemon_wiring(env, repo_root=ROOT)
    if not wiring.enabled:
        print("LKW hybrid daemon wiring must be enabled", file=sys.stderr)
        return 1
    if wiring.spec is None or wiring.launcher_path is None:
        print("LKW hybrid daemon spec or launcher missing", file=sys.stderr)
        return 1
    if wiring.spec.process_name != "lkw-host":
        print("LKW hybrid daemon must use lkw-host process name", file=sys.stderr)
        return 1
    print(f"OK: LKW hybrid daemon ({wiring.spec.bind_host}:{wiring.spec.port})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
